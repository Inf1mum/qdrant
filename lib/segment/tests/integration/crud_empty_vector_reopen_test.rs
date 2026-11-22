//! A named vector added via CRUD to a non-appendable segment holds no alive
//! vectors. Before a restart it is backed by the in-memory `EmptyDense`
//! placeholder, but reopening the segment used to materialize a real chunked
//! mmap storage instead: chunk files preallocated to `CHUNK_SIZE` on disk and
//! one zeroed, deleted slot per point faulted into RAM by the
//! `InRamChunkedMmap` populate-on-open.
//!
//! https://github.com/qdrant/qdrant/issues/10489

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use common::ambient;
use rand::SeedableRng;
use rand::prelude::StdRng;
use segment::data_types::named_vectors::NamedVectors;
use segment::data_types::vector_name_config::{
    DenseVectorConfig, DenseVectorNameConfig, VectorNameConfig,
};
use segment::entry::entry_point::{NonAppendableSegmentEntry as _, SegmentEntry};
use segment::fixtures::payload_fixtures::random_vector;
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::{build_segment, load_segment};
use segment::types::{
    Distance, HnswGlobalConfig, Indexes, SegmentConfig, VectorDataConfig, VectorStorageType,
};
use segment::vector_storage::VectorStorageRead as _;
use tempfile::Builder;
use uuid::Uuid;

const VECTOR: &str = "main";
const EXTRA: &str = "extra";
const DIM: usize = 128;
const POINTS: usize = 1000;

/// Appendable source segment with `POINTS` points in `VECTOR`.
fn source_config() -> SegmentConfig {
    SegmentConfig {
        vector_data: HashMap::from([(
            VECTOR.to_owned(),
            VectorDataConfig {
                size: DIM,
                distance: Distance::Dot,
                storage_type: VectorStorageType::ChunkedMmap,
                index: Indexes::Plain {},
                quantization_config: None,
                multivector_config: None,
                datatype: None,
            },
        )]),
        sparse_vector_data: Default::default(),
        payload_storage_type: Default::default(),
        id_tracker_memory: None,
    }
}

/// Indexed (non-appendable) destination config, like after optimization.
fn indexed_config() -> SegmentConfig {
    SegmentConfig {
        vector_data: HashMap::from([(
            VECTOR.to_owned(),
            VectorDataConfig {
                size: DIM,
                distance: Distance::Dot,
                storage_type: VectorStorageType::default(),
                index: Indexes::Hnsw(Default::default()),
                quantization_config: None,
                multivector_config: None,
                datatype: None,
            },
        )]),
        sparse_vector_data: Default::default(),
        payload_storage_type: Default::default(),
        id_tracker_memory: None,
    }
}

/// Builds an appendable segment with `POINTS` points, then optimizes it into
/// an indexed, non-appendable segment.
fn build_indexed_segment(dir: &Path, temp_dir: &Path) -> Segment {
    let stopped = AtomicBool::new(false);

    let (mut segment, token) = build_segment(dir, &source_config(), None, true).unwrap();
    let _ = token;

    let _scope = ambient::test_guard();
    let mut rng = StdRng::seed_from_u64(42);

    for idx in 0..POINTS as u64 {
        let vector: Vec<f32> = random_vector(&mut rng, DIM);
        segment
            .upsert_point(
                1 + idx,
                idx.into(),
                NamedVectors::from_pairs([(VECTOR.to_owned(), vector)]),
            )
            .unwrap();
    }

    let mut builder = SegmentBuilder::new(
        temp_dir,
        &indexed_config(),
        &HnswGlobalConfig::default(),
        common::flags::FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&segment], &stopped).unwrap();

    let built = builder.build_for_test(dir);
    assert!(
        !built.appendable_flag,
        "indexed segment must be non-appendable",
    );
    built
}

fn chunk_files(segment_path: &Path) -> Vec<std::path::PathBuf> {
    let vectors_dir = segment_path
        .join(format!("vector_storage-{EXTRA}"))
        .join("vectors");
    let mut files: Vec<std::path::PathBuf> = match fs_err::read_dir(&vectors_dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("chunk_"))
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    files
}

/// Loads the segment at `path` twice, mimicking the issue's restart flow, and
/// checks the "extra" vector storage stays a placeholder with no files on
/// disk. Returns the second reload.
fn assert_reopen_stays_placeholder(segment_path: &Path) -> Segment {
    let stopped = AtomicBool::new(false);
    let uuid = Uuid::nil();

    // The issue's RAM cost is paid on every restart, so reload twice: the
    // first reopen must not materialize the storage, the second must not
    // pre-fault anything that the first one (wrongly) left behind.
    let _first = load_segment(segment_path, uuid, None, &stopped, false).unwrap();
    let reloaded = load_segment(segment_path, uuid, None, &stopped, false).unwrap();

    let chunks = chunk_files(segment_path);
    assert!(
        chunks.is_empty(),
        "reopening must not materialize chunk files for the CRUD-created empty vector, found {chunks:?}",
    );

    {
        let storage = reloaded.vector_data[EXTRA].vector_storage.borrow();
        assert!(
            matches!(
                *storage,
                segment::vector_storage::VectorStorageEnum::EmptyDense(_)
            ),
            "reopened storage must stay the empty placeholder, got {storage:?}",
        );
        assert_eq!(storage.total_vector_count(), POINTS);
        assert_eq!(storage.deleted_vector_count(), POINTS);
    }

    reloaded
}

#[test]
fn crud_created_empty_named_vector_reopens_as_placeholder() {
    let dir = Builder::new()
        .prefix("crud_extra_segments")
        .tempdir()
        .unwrap();
    let temp_dir = Builder::new().prefix("crud_extra_temp").tempdir().unwrap();

    let mut built = build_indexed_segment(dir.path(), temp_dir.path());

    // CRUD: add a named vector, no point ever gets it.
    let created = built
        .create_vector_name(
            10_000,
            EXTRA,
            &VectorNameConfig::Dense(DenseVectorNameConfig {
                dense: DenseVectorConfig {
                    size: DIM,
                    distance: Distance::Dot,
                    multivector_config: None,
                    datatype: None,
                },
            }),
        )
        .unwrap();
    assert!(created);

    // Before the restart: in-memory placeholder, nothing on disk.
    let storage = built.vector_data[EXTRA].vector_storage.borrow();
    assert!(matches!(
        *storage,
        segment::vector_storage::VectorStorageEnum::EmptyDense(_)
    ));
    assert_eq!(storage.total_vector_count(), POINTS);
    drop(storage);
    assert!(
        chunk_files(&built.segment_path).is_empty(),
        "CRUD creation must not write chunk files",
    );

    let segment_path = built.segment_path.clone();
    drop(built);

    // After the restart: the storage must remain a placeholder.
    let reloaded = assert_reopen_stays_placeholder(&segment_path);

    // And the segment must still work.
    let _scope = ambient::test_guard();
    let mut rng = StdRng::seed_from_u64(43);
    let query: Vec<f32> = random_vector(&mut rng, DIM);

    let hits = reloaded
        .search(
            VECTOR,
            &query.clone().into(),
            &segment::types::WithPayload::default(),
            &false.into(),
            None,
            5,
            None,
        )
        .unwrap();
    assert_eq!(hits.len(), 5, "search on the original vector must work");

    let hits = reloaded
        .search(
            EXTRA,
            &query.into(),
            &segment::types::WithPayload::default(),
            &false.into(),
            None,
            5,
            None,
        )
        .unwrap();
    assert!(
        hits.is_empty(),
        "no point has the CRUD-added vector, search must return nothing",
    );
}
