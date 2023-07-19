# using Preferences
# set_preferences!(
#     MLIR.API.MLIR_jll,
#     "mlir_c_path" => "/home/jumerckx/julia/build/debug/standalone/lib/libStandaloneCAPITestLib.so",
# )

using MLIR
ctx = MLIR.API.mlirContextCreate()

function registerAllDialects!(ctx)
    registry = MLIR.API.mlirDialectRegistryCreate()
    MLIR.API.mlirRegisterAllDialects(registry)
    handle = MLIR.API.mlirGetDialectHandle__jlir__()
    MLIR.API.mlirDialectHandleRegisterDialect(handle, ctx)
    MLIR.API.mlirContextAppendDialectRegistry(ctx, registry)
    MLIR.API.mlirDialectRegistryDestroy(registry)

    return nothing
end

registerAllDialects!(ctx)

ir = """
pdl.pattern : benefit(1) {
  %root = pdl.operation "test.op"
  pdl.rewrite %root {
    pdl.operation "test.success2"
    pdl.erase %root
  }
}
"""

mod = MLIR.API.mlirModuleCreateParse(ctx, ir)
op = MLIR.API.mlirModuleGetOperation(mod)

MLIR.API.mlirOperationVerify(op)

ir = """
module {
    func.func nested @"Tuple{typeof(Main.branches), Bool}"(%arg0: !jlir<"typeof(Main.branches)">, %arg1: !jlir.Bool) -> !jlir.Bool attributes {llvm.emit_c_interface} {
      "jlir.goto"()[^bb1] : () -> ()
    ^bb1:  // pred: ^bb0
      "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
    ^bb2:  // pred: ^bb1
      %0 = "jlir.pi"(%arg1) : (!jlir.Bool) -> !jlir.Bool
      "jlir.return"(%0) : (!jlir.Bool) -> ()
    ^bb3:  // pred: ^bb1
      %1 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
      "jlir.return"(%1) : (!jlir.Bool) -> ()
    }
}
"""

ir = """
module {
    func.func nested @"Tuple{jlir.Function, Bool}"(%arg0: !jlir.Bool, %arg1: !jlir.Bool) -> !jlir.Bool attributes {llvm.emit_c_interface} {
      "jlir.goto"()[^bb1] : () -> ()
    ^bb1:  // pred: ^bb0
      "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
    ^bb2:  // pred: ^bb1
      %0 = "jlir.pi"(%arg1) : (!jlir.Bool) -> !jlir.Bool
      "jlir.return"(%0) : (!jlir.Bool) -> ()
    ^bb3:  // pred: ^bb1
      %1 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
      "jlir.return"(%1) : (!jlir.Bool) -> ()
    }
}
"""

mod = MLIR.API.mlirModuleCreateParse(ctx, ir)

function lowerModuleToLLVM(ctx, mod)
    pm = MLIR.API.mlirPassManagerCreate(ctx)
    op = "func.func"
    opm = MLIR.API.mlirPassManagerGetNestedUnder(pm, op)
    MLIR.API.mlirPassManagerAddOwnedPass(pm,
        MLIR.API.mlirCreateConversionConvertFuncToLLVM()
    )
    MLIR.API.mlirOpPassManagerAddOwnedPass(opm,
        MLIR.API.mlirCreateConversionConvertArithmeticToLLVM()
    )
    status = MLIR.API.mlirPassManagerRun(pm, mod)
    # undefined symbol: mlirLogicalResultIsFailure
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    MLIR.API.mlirPassManagerDestroy(pm)
end


lowerModuleToLLVM(ctx, mod)

using MLIR: IR

ctx = IR.Context()

IR.get_or_load_dialect!(ctx, "jlir") |> typeof

op = IR.create_operation("jlir.neg_int", IR.Location(ctx));

typeof(op)

op = IR.create_operation("jlir.pi", IR.Location(ctx), results=[typeof(pi)]);

op = IR.create_operation("jlir.checked_sadd_int", IR.Location(ctx), results=[typeof(Int)]);
typeof(op)


op = IR.create_operation("jlir.add_int", IR.Location(ctx));
typeof(op)


using MLIR: IR
ctx = IR.Context()
IR.get_or_load_dialect!(ctx, "jlir");

loc = IR.Location(ctx)

op1 = IR.create_operation("jlir.add_int", loc                        ); # OK
op2 = IR.create_operation("jlir.add_int", loc, results = [Int]       ); # OK
op3 = IR.create_operation("jlir.add_int", loc, results = [String]    ); # OK?
op3 = IR.create_operation("jlir.add_int", loc, operands = [Int, Int] ); # crash

include("./Utils.jl")
using .Utils

function pow(x::F, n::Integer) where {F}
  p = one(F)
  for _ in 1:n
      p *= x
  end
  p
end

ir, ret = @code_ircode pow(2, 10)

println(@dot pow(2, 10))
