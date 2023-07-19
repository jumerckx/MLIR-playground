include("./Utils.jl")

using MLIR
using MLIR: IR, API, API.mlir_c
using .Utils

f(x, y) = 2*(x+y)
ir, ret = @code_ircode f(2, 3)


function registerAllDialects!(ctx)
    registry = MLIR.API.mlirDialectRegistryCreate()
    MLIR.API.mlirRegisterAllDialects(registry)
    handle = MLIR.API.mlirGetDialectHandle__jlir__()
    API.mlirDialectHandleInsertDialect(handle, registry)
    MLIR.API.mlirContextAppendDialectRegistry(ctx, registry)
    # MLIR.API.mlirDialectRegistryDestroy(registry)

    MLIR.API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end

ctx = API.mlirContextCreate()
registry = registerAllDialects!(ctx)

state = Ref(API.mlirOperationStateGet("func.func", API.mlirLocationUnknownGet(ctx)))

function API.MlirType(ctx::API.MlirContext, t)
    return @ccall mlir_c.brutus_get_jlirtype(ctx::API.MlirContext, t::Any)::API.MlirType
end

argtypes = let
    argtypes = getfield(ir, :argtypes)
    API.MlirType.(Ref(ctx), argtypes)
end

reg = API.mlirRegionCreate()
entry_block = API.mlirBlockCreate(length(argtypes), argtypes, [API.mlirLocationUnknownGet(ctx) for _ in enumerate(argtypes)])

API.mlirRegionAppendOwnedBlock(reg, entry_block)
API.mlirOperationStateAddOwnedRegions(state, 1, [reg])

push!(block::API.MlirBlock, type::API.MlirType, loc::API.MlirLocation) =
    API.mlirBlockAddArgument(block, type, loc)

input_types = API.mlirBlockGetArgument.(Ref(entry_block), eachindex(argtypes) .- 1)

API.mlirBlockGetNumArguments(entry_block)

push!(block::API.MlirBlock, op::API.MlirOperation) = 
    API.mlirBlockAppendOwnedOperation(block, op)

add_op = IR.create_operation("jlir.add_int", API.mlirLocationUnknownGet(ctx); operands=[API.mlirBlockGetArgument(entry_block, 1), API.mlirBlockGetArgument(entry_block, 2)]) # "jlir.add_int"(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64

named_val_attr = let 
    val_attr = @ccall mlir_c.brutus_get_jlirattr(ctx::API.MlirContext, 2::Any)::API.MlirAttribute
    API.mlirNamedAttributeGet(API.mlirIdentifierGet(ctx, "value"), val_attr)
end

constant_op = IR.create_operation("jlir.constant", API.mlirLocationUnknownGet(ctx); attributes=[named_val_attr], results=[API.MlirType(ctx, Int)]) # "jlir.constant"() {value = #jlir<2>} : () -> !jlir.Int64

mul_op = IR.create_operation("jlir.mul_int", API.mlirLocationUnknownGet(ctx); operands=[API.mlirOperationGetResult(constant_op, 0), API.mlirOperationGetResult(add_op, 0)])

ret_op = IR.create_operation("func.return", API.mlirLocationUnknownGet(ctx); operands=[API.mlirBlockGetArgument(entry_block, 2)], result_inference = false)

push!(entry_block, add_op)
push!(entry_block, constant_op)
push!(entry_block, mul_op)
push!(entry_block, ret_op)

named_type_attr = let
    function_type = API.mlirFunctionTypeGet(
        ctx,
        length(input_types), API.mlirValueGetType.(input_types),
        1, [API.MlirType(ctx, ret)])
        
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

function push!(state::Base.RefValue{MLIR.API.MlirOperationState}, attr::IR.MlirNamedAttribute)
    API.mlirOperationStateAddAttributes(state, 1, Ref(attr))
end

push!(state, named_type_attr)
push!(state, named_symbol_name_attr)
push!(state, named_viz_attr)
push!(state, named_unit_attr)

op = API.mlirOperationCreate(state)

API.mlirOperationVerify(op)

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

@show op
