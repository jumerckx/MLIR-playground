include("./Utils.jl")

using MLIR
using MLIR: IR, API, API.mlir_c
using .Utils

f(x, y) = 2*(x+y) # simple function with only one block.
ir, ret = @code_ircode f(2, 3)
#=
7 1 ─ %1 = Base.add_int(_2, _3)::Int64 │╻ +
  │   %2 = Base.mul_int(2, %1)::Int64  │╻ *
  └──      return %2                   │ 
=#

function registerAllDialects!(ctx)
    registry = MLIR.API.mlirDialectRegistryCreate()
    MLIR.API.mlirRegisterAllDialects(registry)
    handle = MLIR.API.mlirGetDialectHandle__jlir__()
    API.mlirDialectHandleInsertDialect(handle, registry)
    MLIR.API.mlirContextAppendDialectRegistry(ctx, registry)
    MLIR.API.mlirDialectRegistryDestroy(registry)

    MLIR.API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end

ctx = API.mlirContextCreate()
registry = registerAllDialects!(ctx)

### Create operation scaffolding ###
state = Ref(API.mlirOperationStateGet("func.func", API.mlirLocationUnknownGet(ctx)))

argtypes = let
    argtypes = getfield(ir, :argtypes)
    API.brutus_get_jlirtype.(Ref(ctx), argtypes)
end

reg = API.mlirRegionCreate()
entry_block = API.mlirBlockCreate(length(argtypes), argtypes, [API.mlirLocationUnknownGet(ctx) for _ in enumerate(argtypes)])

API.mlirRegionAppendOwnedBlock(reg, entry_block) # simple function only has one block, otherwise, more blocks would need to be added.
API.mlirOperationStateAddOwnedRegions(state, 1, [reg])

Base.push!(block::API.MlirBlock, type::API.MlirType, loc::API.MlirLocation) =
    API.mlirBlockAddArgument(block, type, loc)

API.mlirBlockGetNumArguments(entry_block)

### Add statements to block ###
Base.push!(block::API.MlirBlock, op::API.MlirOperation) = 
    API.mlirBlockAppendOwnedOperation(block, op)

add_op = IR.create_operation("jlir.add_int", API.mlirLocationUnknownGet(ctx); operands=[API.mlirBlockGetArgument(entry_block, 1), API.mlirBlockGetArgument(entry_block, 2)]) # "jlir.add_int"(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
@atomic add_op.owned = false
add_op = add_op.operation

named_val_attr = let 
    val_attr = @ccall mlir_c.brutus_get_jlirattr(ctx::API.MlirContext, 2::Any)::API.MlirAttribute
    API.mlirNamedAttributeGet(API.mlirIdentifierGet(ctx, "value"), val_attr)
end
constant_op = IR.create_operation("jlir.constant", API.mlirLocationUnknownGet(ctx); attributes=[named_val_attr], results=[API.brutus_get_jlirtype(ctx, Int)]) # "jlir.constant"() {value = #jlir<2>} : () -> !jlir.Int64
@atomic constant_op.owned = false
constant_op = constant_op.operation

mul_op = IR.create_operation("jlir.mul_int", API.mlirLocationUnknownGet(ctx); operands=[API.mlirOperationGetResult(constant_op, 0), API.mlirOperationGetResult(add_op, 0)])
@atomic mul_op.owned = false
mul_op = mul_op.operation

####################
IR.create_operation("jlir.mul_int", API.mlirLocationUnknownGet(ctx); operands=[API.mlirOperationGetResult(constant_op, 0), API.mlirOperationGetResult(add_op, 0)])
API.brutus_get_jlirtype(isa(ctx, IR.Context) ? ctx.context : ctx, Int)
IR.get_result(IR.create_operation("jlir.mul_int", API.mlirLocationUnknownGet(ctx); operands=[API.mlirOperationGetResult(constant_op, 0), API.mlirOperationGetResult(add_op, 0)]), 1).value |> API.mlirValueGetType
####################


# ret_op = IR.create_operation("func.return", API.mlirLocationUnknownGet(ctx); operands=[API.mlirBlockGetArgument(entry_block, 2)], result_inference = false)
ret_op = IR.create_operation("func.return", API.mlirLocationUnknownGet(ctx); operands=[API.mlirOperationGetResult(mul_op, 0)], result_inference = false)
@atomic ret_op.owned = false
ret_op = ret_op.operation


push!(entry_block, add_op)
push!(entry_block, constant_op)
push!(entry_block, mul_op)
push!(entry_block, ret_op)

### Add attributes to top-level operation "func.func" ###
input_values = API.mlirBlockGetArgument.(Ref(entry_block), eachindex(argtypes) .- 1)

named_type_attr = let
    function_type = API.mlirFunctionTypeGet(
        ctx,
        length(input_values), API.mlirValueGetType.(input_values),
        1, [API.brutus_get_jlirtype(ctx, ret)])
        
    type_attr = API.mlirTypeAttrGet(function_type)
    
    API.mlirNamedAttributeGet(API.mlirIdentifierGet(ctx, "function_type"), type_attr)
end

named_symbol_name_attr = let 
    name = "f"
    
    symbol_name_attr = API.mlirStringAttrGet(ctx, name)
    
    API.mlirNamedAttributeGet(API.mlirIdentifierGet(ctx, "sym_name"), symbol_name_attr)
end

named_viz_attr = let
    viz_attr = API.mlirStringAttrGet(ctx, "nested")
    
    API.mlirNamedAttributeGet(API.mlirIdentifierGet(ctx, "sym_visibility"), viz_attr)
end

named_unit_attr = let 
    unit_attr = API.mlirUnitAttrGet(ctx)

    API.mlirNamedAttributeGet(API.mlirIdentifierGet(ctx, "llvm.emit_c_interface"), unit_attr)
end

function Base.push!(state::Base.RefValue{MLIR.API.MlirOperationState}, attr::IR.MlirNamedAttribute)
    API.mlirOperationStateAddAttributes(state, 1, Ref(attr))
end

push!(state, named_type_attr)
push!(state, named_symbol_name_attr)
push!(state, named_viz_attr)
push!(state, named_unit_attr)

### Create final operation and verify ###
op = API.mlirOperationCreate(state);

if (API.mlirOperationVerify(op)); @info "Operation was succesfully verified!"; end

function print_callback(str::API.MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

function Base.show(io::IO, operation::API.MlirOperation)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    flags = API.mlirOpPrintingFlagsCreate()
    get(io, :debug, false) && API.mlirOpPrintingFlagsEnableDebugInfo(flags, true, true)
    API.mlirOperationPrintWithFlags(operation, flags, c_print_callback, ref)
    println(io)
end

op
test = IR.Operation(op)
@show test
#=
func.func nested @f(%arg0: !jlir<typeof(Main.f)>, %arg1: !jlir.Int64, %arg2: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
  %0 = "jlir.add_int"(%arg1, %arg2) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
  %1 = "jlir.constant"() {value = #jlir<2>} : () -> !jlir.Int64
  %2 = "jlir.mul_int"(%1, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
  return %arg2 : !jlir.Int64
}
=#