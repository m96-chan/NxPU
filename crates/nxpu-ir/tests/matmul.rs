//! Integration test: build a vector-add + matmul-like IR module programmatically
//! and verify the text dump output.

use nxpu_ir::*;

/// Build a simple vector addition IR:
///
/// ```wgsl
/// @group(0) @binding(0) var<storage, read> a: array<f32>;
/// @group(0) @binding(1) var<storage, read> b: array<f32>;
/// @group(0) @binding(2) var<storage, read_write> result: array<f32>;
///
/// @compute @workgroup_size(256)
/// fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
///     let i = gid.x;
///     result[i] = a[i] + b[i];
/// }
/// ```
#[test]
fn build_vector_add_module() {
    let mut module = Module::default();

    // ---- Types ----
    let f32_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::F32),
    });
    let _u32_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::U32),
    });
    let vec3u_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Vector {
            size: VectorSize::Tri,
            scalar: Scalar::U32,
        },
    });
    let array_f32_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Array {
            base: f32_ty,
            size: ArraySize::Dynamic,
            stride: 4,
        },
    });

    // Verify type deduplication
    let f32_ty2 = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::F32),
    });
    assert_eq!(f32_ty, f32_ty2, "f32 type should be deduplicated");

    // ---- Global Variables ----
    let gv_a = module.global_variables.append(GlobalVariable {
        name: Some("a".into()),
        space: AddressSpace::Storage {
            access: StorageAccess::LOAD,
        },
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
        ty: array_f32_ty,
        init: None,
        layout: None,
    });
    let gv_b = module.global_variables.append(GlobalVariable {
        name: Some("b".into()),
        space: AddressSpace::Storage {
            access: StorageAccess::LOAD,
        },
        binding: Some(ResourceBinding {
            group: 0,
            binding: 1,
        }),
        ty: array_f32_ty,
        init: None,
        layout: None,
    });
    let gv_result = module.global_variables.append(GlobalVariable {
        name: Some("result".into()),
        space: AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        },
        binding: Some(ResourceBinding {
            group: 0,
            binding: 2,
        }),
        ty: array_f32_ty,
        init: None,
        layout: None,
    });

    // ---- Entry Point Function ----
    let mut function = Function::new("main");
    function.arguments.push(FunctionArgument {
        name: Some("gid".into()),
        ty: vec3u_ty,
        binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
    });

    // Build expressions
    let emit_start = function.expressions.next_handle();

    // [0] gid (function argument 0)
    let gid_expr = function.expressions.append(Expression::FunctionArgument(0));
    // [1] gid.x
    let gid_x = function.expressions.append(Expression::AccessIndex {
        base: gid_expr,
        index: 0,
    });
    // [2] &a (global variable pointer)
    let a_ptr = function
        .expressions
        .append(Expression::GlobalVariable(gv_a));
    // [3] &b
    let b_ptr = function
        .expressions
        .append(Expression::GlobalVariable(gv_b));
    // [4] &result
    let result_ptr = function
        .expressions
        .append(Expression::GlobalVariable(gv_result));
    // [5] a[gid.x] (pointer)
    let a_elem_ptr = function.expressions.append(Expression::Access {
        base: a_ptr,
        index: gid_x,
    });
    // [6] b[gid.x] (pointer)
    let b_elem_ptr = function.expressions.append(Expression::Access {
        base: b_ptr,
        index: gid_x,
    });
    // [7] result[gid.x] (pointer)
    let result_elem_ptr = function.expressions.append(Expression::Access {
        base: result_ptr,
        index: gid_x,
    });
    // [8] load a[gid.x]
    let a_val = function.expressions.append(Expression::Load {
        pointer: a_elem_ptr,
    });
    // [9] load b[gid.x]
    let b_val = function.expressions.append(Expression::Load {
        pointer: b_elem_ptr,
    });
    // [10] a[gid.x] + b[gid.x]
    let sum = function.expressions.append(Expression::Binary {
        op: BinaryOp::Add,
        left: a_val,
        right: b_val,
    });

    let emit_end = function.expressions.next_handle();

    // Name some expressions for debugging
    function.named_expressions.insert(gid_x, "i".into());
    function.named_expressions.insert(sum, "sum".into());

    // Build body
    function
        .body
        .push(Statement::Emit(Range::new(emit_start, emit_end)));
    function.body.push(Statement::Store {
        pointer: result_elem_ptr,
        value: sum,
    });

    // Add entry point
    module.entry_points.push(EntryPoint {
        name: "main".into(),
        workgroup_size: [256, 1, 1],
        function,
    });

    // ---- Verify ----
    assert_eq!(module.types.len(), 4);
    assert_eq!(module.global_variables.len(), 3);
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);

    // Dump and check key patterns
    let dump = dump_module(&module);
    assert!(dump.contains("f32"), "dump should contain f32 type");
    assert!(dump.contains("u32"), "dump should contain u32 type");
    assert!(dump.contains("vec3<u32>"), "dump should contain vec3<u32>");
    assert!(
        dump.contains("array<f32>"),
        "dump should contain array<f32>"
    );
    assert!(
        dump.contains("@group(0) @binding(0)"),
        "dump should contain binding annotation"
    );
    assert!(
        dump.contains("@group(0) @binding(2)"),
        "dump should contain binding(2)"
    );
    assert!(
        dump.contains("workgroup_size(256, 1, 1)"),
        "dump should contain workgroup_size"
    );
    assert!(
        dump.contains("@builtin(global_invocation_id)"),
        "dump should contain builtin binding"
    );
    assert!(dump.contains("fn main"), "dump should contain fn main");
    assert!(
        dump.contains("Store"),
        "dump should contain Store statement"
    );

    // Print the dump for manual inspection
    eprintln!("{dump}");
}

/// Build a simplified matmul IR with a loop:
///
/// ```wgsl
/// @group(0) @binding(0) var<storage, read> a: array<f32>;
/// @group(0) @binding(1) var<storage, read> b: array<f32>;
/// @group(0) @binding(2) var<storage, read_write> c: array<f32>;
///
/// const N: u32 = 64;
///
/// @compute @workgroup_size(64)
/// fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
///     let row = gid.x / N;
///     let col = gid.x % N;
///     var sum: f32 = 0.0;
///     for (var i: u32 = 0u; i < N; i++) {
///         sum += a[row * N + i] * b[i * N + col];
///     }
///     c[gid.x] = sum;
/// }
/// ```
#[test]
fn build_matmul_module() {
    let mut module = Module::default();

    // Types
    let f32_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::F32),
    });
    let u32_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::U32),
    });
    let vec3u_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Vector {
            size: VectorSize::Tri,
            scalar: Scalar::U32,
        },
    });
    let array_f32_ty = module.types.insert(Type {
        name: None,
        inner: TypeInner::Array {
            base: f32_ty,
            size: ArraySize::Dynamic,
            stride: 4,
        },
    });

    // Global variables
    let gv_a = module.global_variables.append(GlobalVariable {
        name: Some("a".into()),
        space: AddressSpace::Storage {
            access: StorageAccess::LOAD,
        },
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
        ty: array_f32_ty,
        init: None,
        layout: None,
    });
    let gv_b = module.global_variables.append(GlobalVariable {
        name: Some("b".into()),
        space: AddressSpace::Storage {
            access: StorageAccess::LOAD,
        },
        binding: Some(ResourceBinding {
            group: 0,
            binding: 1,
        }),
        ty: array_f32_ty,
        init: None,
        layout: None,
    });
    let gv_c = module.global_variables.append(GlobalVariable {
        name: Some("c".into()),
        space: AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        },
        binding: Some(ResourceBinding {
            group: 0,
            binding: 2,
        }),
        ty: array_f32_ty,
        init: None,
        layout: None,
    });

    // Entry point function
    let mut function = Function::new("main");
    function.arguments.push(FunctionArgument {
        name: Some("gid".into()),
        ty: vec3u_ty,
        binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
    });

    // Local variable: var sum: f32 = 0.0
    let sum_init = function
        .expressions
        .append(Expression::Literal(Literal::F32(0.0)));
    let sum_var = function.local_variables.append(LocalVariable {
        name: Some("sum".into()),
        ty: f32_ty,
        init: Some(sum_init),
    });

    // Local variable: var i: u32 = 0u
    let i_init = function
        .expressions
        .append(Expression::Literal(Literal::U32(0)));
    let i_var = function.local_variables.append(LocalVariable {
        name: Some("i".into()),
        ty: u32_ty,
        init: Some(i_init),
    });

    // Build expressions for the main body
    let emit1_start = function.expressions.next_handle();

    // gid argument
    let gid_expr = function.expressions.append(Expression::FunctionArgument(0));
    // gid.x
    let gid_x = function.expressions.append(Expression::AccessIndex {
        base: gid_expr,
        index: 0,
    });
    // N = 64u
    let n_val = function
        .expressions
        .append(Expression::Literal(Literal::U32(64)));
    // row = gid.x / N
    let row = function.expressions.append(Expression::Binary {
        op: BinaryOp::Divide,
        left: gid_x,
        right: n_val,
    });
    // col = gid.x % N
    let col = function.expressions.append(Expression::Binary {
        op: BinaryOp::Modulo,
        left: gid_x,
        right: n_val,
    });
    // &sum
    let sum_ptr = function
        .expressions
        .append(Expression::LocalVariable(sum_var));
    // &i
    let i_ptr = function
        .expressions
        .append(Expression::LocalVariable(i_var));

    // Global variable pointers
    let a_ptr = function
        .expressions
        .append(Expression::GlobalVariable(gv_a));
    let b_ptr = function
        .expressions
        .append(Expression::GlobalVariable(gv_b));
    let c_ptr = function
        .expressions
        .append(Expression::GlobalVariable(gv_c));

    let emit1_end = function.expressions.next_handle();

    // --- Loop body expressions ---
    let emit_loop_start = function.expressions.next_handle();

    // load i
    let i_val = function
        .expressions
        .append(Expression::Load { pointer: i_ptr });
    // row * N
    let row_times_n = function.expressions.append(Expression::Binary {
        op: BinaryOp::Multiply,
        left: row,
        right: n_val,
    });
    // row * N + i
    let a_idx = function.expressions.append(Expression::Binary {
        op: BinaryOp::Add,
        left: row_times_n,
        right: i_val,
    });
    // i * N
    let i_times_n = function.expressions.append(Expression::Binary {
        op: BinaryOp::Multiply,
        left: i_val,
        right: n_val,
    });
    // i * N + col
    let b_idx = function.expressions.append(Expression::Binary {
        op: BinaryOp::Add,
        left: i_times_n,
        right: col,
    });
    // a[row*N+i] (pointer)
    let a_elem_ptr = function.expressions.append(Expression::Access {
        base: a_ptr,
        index: a_idx,
    });
    // b[i*N+col] (pointer)
    let b_elem_ptr = function.expressions.append(Expression::Access {
        base: b_ptr,
        index: b_idx,
    });
    // load a[...]
    let a_val = function.expressions.append(Expression::Load {
        pointer: a_elem_ptr,
    });
    // load b[...]
    let b_val = function.expressions.append(Expression::Load {
        pointer: b_elem_ptr,
    });
    // a[...] * b[...]
    let product = function.expressions.append(Expression::Binary {
        op: BinaryOp::Multiply,
        left: a_val,
        right: b_val,
    });
    // load sum
    let sum_val = function
        .expressions
        .append(Expression::Load { pointer: sum_ptr });
    // sum + product
    let new_sum = function.expressions.append(Expression::Binary {
        op: BinaryOp::Add,
        left: sum_val,
        right: product,
    });

    let emit_loop_end = function.expressions.next_handle();

    // --- Continuing block expressions ---
    let emit_cont_start = function.expressions.next_handle();

    // load i (for increment)
    let i_val2 = function
        .expressions
        .append(Expression::Load { pointer: i_ptr });
    // 1u
    let one = function
        .expressions
        .append(Expression::Literal(Literal::U32(1)));
    // i + 1
    let i_plus_one = function.expressions.append(Expression::Binary {
        op: BinaryOp::Add,
        left: i_val2,
        right: one,
    });
    // i < N (break condition: !(i < N) means i >= N)
    let i_ge_n = function.expressions.append(Expression::Binary {
        op: BinaryOp::GreaterEqual,
        left: i_plus_one,
        right: n_val,
    });

    let emit_cont_end = function.expressions.next_handle();

    // --- Post-loop expressions ---
    let emit_post_start = function.expressions.next_handle();

    // load sum (final)
    let final_sum = function
        .expressions
        .append(Expression::Load { pointer: sum_ptr });
    // c[gid.x] pointer
    let c_elem_ptr = function.expressions.append(Expression::Access {
        base: c_ptr,
        index: gid_x,
    });

    let emit_post_end = function.expressions.next_handle();

    // --- Build statements ---

    // Loop body: emit, store sum
    let loop_body = vec![
        Statement::Emit(Range::new(emit_loop_start, emit_loop_end)),
        Statement::Store {
            pointer: sum_ptr,
            value: new_sum,
        },
    ];

    // Continuing: i = i + 1, break_if i >= N
    let continuing = vec![
        Statement::Emit(Range::new(emit_cont_start, emit_cont_end)),
        Statement::Store {
            pointer: i_ptr,
            value: i_plus_one,
        },
    ];

    // Main body
    function
        .body
        .push(Statement::Emit(Range::new(emit1_start, emit1_end)));
    function.body.push(Statement::Loop {
        body: loop_body,
        continuing,
        break_if: Some(i_ge_n),
    });
    function
        .body
        .push(Statement::Emit(Range::new(emit_post_start, emit_post_end)));
    function.body.push(Statement::Store {
        pointer: c_elem_ptr,
        value: final_sum,
    });

    // Name key expressions
    function.named_expressions.insert(row, "row".into());
    function.named_expressions.insert(col, "col".into());

    module.entry_points.push(EntryPoint {
        name: "main".into(),
        workgroup_size: [64, 1, 1],
        function,
    });

    // ---- Verify ----
    assert_eq!(module.types.len(), 4); // f32, u32, vec3<u32>, array<f32>
    assert_eq!(module.global_variables.len(), 3);
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);

    let ep_func = &module.entry_points[0].function;
    assert_eq!(ep_func.local_variables.len(), 2); // sum, i

    // Verify the dump contains expected patterns
    let dump = dump_module(&module);
    assert!(dump.contains("workgroup_size(64, 1, 1)"));
    assert!(dump.contains("Loop"));
    assert!(dump.contains("Store"));
    assert!(dump.contains("fn main"));

    eprintln!("{dump}");
}

/// Verify that all type variants can be created and stored.
#[test]
fn all_type_variants() {
    let mut types = UniqueArena::new();

    let scalar = types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::F32),
    });
    let vector = types.insert(Type {
        name: None,
        inner: TypeInner::Vector {
            size: VectorSize::Quad,
            scalar: Scalar::F32,
        },
    });
    let matrix = types.insert(Type {
        name: None,
        inner: TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Quad,
            scalar: Scalar::F32,
        },
    });
    let atomic = types.insert(Type {
        name: None,
        inner: TypeInner::Atomic(Scalar::U32),
    });
    let pointer = types.insert(Type {
        name: None,
        inner: TypeInner::Pointer {
            base: scalar,
            space: AddressSpace::Function,
        },
    });
    let array = types.insert(Type {
        name: None,
        inner: TypeInner::Array {
            base: scalar,
            size: ArraySize::Constant(16),
            stride: 4,
        },
    });
    let _struct_ty = types.insert(Type {
        name: Some("MyStruct".into()),
        inner: TypeInner::Struct {
            members: vec![
                StructMember {
                    name: Some("x".into()),
                    ty: scalar,
                    offset: 0,
                },
                StructMember {
                    name: Some("y".into()),
                    ty: vector,
                    offset: 4,
                },
            ],
            span: 20,
        },
    });

    // All 7 variants + all unique
    assert_eq!(types.len(), 7);

    // Verify formatting
    assert_eq!(format_type(&types[scalar], &types), "f32");
    assert_eq!(format_type(&types[vector], &types), "vec4<f32>");
    assert_eq!(format_type(&types[matrix], &types), "mat4x4<f32>");
    assert_eq!(format_type(&types[atomic], &types), "atomic<u32>");
    assert_eq!(format_type(&types[pointer], &types), "ptr<function, f32>");
    assert_eq!(
        format_type(&types[array], &types),
        "array<f32, 16> /*stride 4*/"
    );
}

/// Verify Module::default() produces a valid empty module.
#[test]
fn empty_module() {
    let module = Module::default();
    assert!(module.types.is_empty());
    assert!(module.global_variables.is_empty());
    assert!(module.global_expressions.is_empty());
    assert!(module.functions.is_empty());
    assert!(module.entry_points.is_empty());

    let dump = dump_module(&module);
    assert!(dump.contains("Types:"));
    assert!(!dump.contains("Global Variables:"));
    assert!(!dump.contains("Functions:"));
    assert!(!dump.contains("Entry Points:"));
}
