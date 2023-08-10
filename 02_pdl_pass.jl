using MLIR.API

op = include("01_CAPI_operation.jl");
#=
func.func nested @f(%arg0: !jlir<typeof(Main.f)>, %arg1: !jlir.Int64, %arg2: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
  %0 = "jlir.add_int"(%arg1, %arg2) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
  %1 = "jlir.constant"() {value = #jlir<2>} : () -> !jlir.Int64
  %2 = "jlir.mul_int"(%1, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
  return %arg2 : !jlir.Int64
}
=#

begin
  ir = """
  pdl.pattern : benefit(1) {
    %resultType = pdl.type
    %inputOperand = pdl.operand
    %secondInputOperand = pdl.operand
    %root = pdl.operation "jlir.add_int"(%inputOperand, %secondInputOperand : !pdl.value, !pdl.value) -> (%resultType : !pdl.type)
    pdl.rewrite %root {
      %newMul = pdl.operation "jlir.mul_int"(%inputOperand, %secondInputOperand : !pdl.value, !pdl.value) -> (%resultType : !pdl.type)
      pdl.replace %root with %newMul
    }
  }
  """
  
  pattern_module = MLIR.API.mlirModuleCreateParse(ctx, ir)
end

pdl_pattern = API.beaverPDLPatternGet(pattern_module)
pattern_set = API.beaverRewritePatternSetGet(ctx)
pattern_set = API.beaverPatternSetAddOwnedPDLPattern(pattern_set, pdl_pattern)

API.beaverApplyOwnedPatternSetOnOperation(op, pattern_set)

op
#=
func.func nested @f(%arg0: !jlir<typeof(Main.f)>, %arg1: !jlir.Int64, %arg2: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
  %0 = "jlir.mul_int"(%arg1, %arg2) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
  %1 = "jlir.constant"() {value = #jlir<2>} : () -> !jlir.Int64
  %2 = "jlir.mul_int"(%1, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
  return %2 : !jlir.Int64
}
=#
