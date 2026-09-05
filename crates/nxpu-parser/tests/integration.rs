//! Integration tests for the WGSL parser.

use nxpu_ir::{AddressSpace, StorageAccess, dump_module};
use nxpu_parser::parse;

#[test]
fn parse_matmul() {
    let source = include_str!("../../../examples/matmul.wgsl");
    let module = parse(source).expect("matmul.wgsl should parse");
    let dump = dump_module(&module);

    // Entry point
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "main");
    assert_eq!(module.entry_points[0].workgroup_size, [16, 16, 1]);

    // Global variables: a, b, result, params
    assert_eq!(module.global_variables.len(), 4);

    // Verify the dump contains expected elements.
    assert!(
        dump.contains("Entry Points:"),
        "dump should have entry points"
    );
    assert!(dump.contains("@compute @workgroup_size(16, 16, 1)"));
    assert!(dump.contains("Global Variables:"));

    // Verify types: should have f32, u32, vec3<u32>, array<f32>, Params struct, etc.
    assert!(module.types.len() >= 4, "should have at least 4 types");
}

#[test]
fn parse_vecadd() {
    let source = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    c[i] = a[i] + b[i];
}
"#;
    let module = parse(source).expect("vecadd should parse");

    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "main");
    assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
    assert_eq!(module.global_variables.len(), 3);

    // Check address spaces.
    for (_, var) in module.global_variables.iter() {
        match var.space {
            AddressSpace::Storage { access } => {
                assert!(
                    access.contains(StorageAccess::LOAD) || access.contains(StorageAccess::STORE)
                );
            }
            _ => panic!("expected storage address space"),
        }
    }
}

#[test]
fn parse_workgroup_barrier() {
    let source = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
var<workgroup> shmem: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    shmem[lid] = buf[lid];
    workgroupBarrier();
    buf[lid] = shmem[lid];
}
"#;
    let module = parse(source).expect("barrier shader should parse");
    assert_eq!(module.entry_points.len(), 1);

    // Should have a workgroup-space global variable.
    let has_workgroup = module
        .global_variables
        .iter()
        .any(|(_, v)| matches!(v.space, AddressSpace::Workgroup));
    assert!(has_workgroup, "should have a workgroup variable");
}

#[test]
fn parse_rejects_fragment_shader() {
    // This is a vertex/fragment shader — should have no compute entry points.
    let source = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
    let module = parse(source).expect("should parse without error");
    // No compute entry points should be extracted.
    assert!(module.entry_points.is_empty());
}

#[test]
fn parse_wgsl_syntax_error() {
    let source = "this is not valid wgsl @@@ {{{";
    let result = parse(source);
    assert!(result.is_err());
}

#[test]
fn parse_empty_compute() {
    let source = r#"
@compute @workgroup_size(1)
fn main() {}
"#;
    let module = parse(source).expect("empty compute should parse");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [1, 1, 1]);
}
