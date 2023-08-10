using MLIR: IR, API

function registerAllDialects!(ctx)
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    handle = API.mlirGetDialectHandle__jlir__()
    API.mlirDialectHandleInsertDialect(handle, registry)
    API.mlirContextAppendDialectRegistry(ctx, registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end

ctx = IR.Context()
registerAllDialects!(ctx);

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
    pattern_module = IR.parse(IR.MModule, ctx, ir)
end

pattern_op = IR.get_operation(pattern_module)

@show pattern_op

manual_pattern_op = let
    op_0 = IR.create_operation("pdl.type", IR.Location(ctx); operands=[], results=[API.mlirTypeParseGet(ctx, "!pdl.type")])
    op_1 = IR.create_operation("pdl.operand", IR.Location(ctx); operands=[], results=[API.mlirTypeParseGet(ctx, "!pdl.value")])
    op_2 = IR.create_operation("pdl.operand", IR.Location(ctx); operands=[], results=[API.mlirTypeParseGet(ctx, "!pdl.value")])
    op_3 = IR.create_operation(
        "pdl.operation",
        IR.Location(ctx);
        operands=IR.get_result.([op_1, op_2, op_0]),
        results=[API.mlirTypeParseGet(ctx, "!pdl.operation")],
        attributes=[
            IR.NamedAttribute(ctx, "attributeValueNames", parse(IR.Attribute, ctx, "[]")),
            IR.NamedAttribute(ctx, "odsOperandSegmentSizes", IR.Attribute(API.mlirDenseI32ArrayGet(ctx, 3, Int32[2, 0, 1]))),
            IR.NamedAttribute(ctx, "opName", IR.Attribute(ctx, "jlir.add_int"))])

    op_4 = IR.create_operation("pdl.operation",
        IR.Location(ctx);
        operands=IR.get_result.([op_1, op_2, op_0]),
        results=[API.mlirTypeParseGet(ctx, "!pdl.operation")],
        attributes=[
            IR.NamedAttribute(ctx, "attributeValueNames", parse(IR.Attribute, ctx, "[]")),
            IR.NamedAttribute(ctx, "odsOperandSegmentSizes", IR.Attribute(API.mlirDenseI32ArrayGet(ctx, 3, Int32[2, 0, 1]))),
            IR.NamedAttribute(ctx, "opName", IR.Attribute(ctx, "jlir.mul_int"))])
    replace_op = IR.create_operation("pdl.replace",
        IR.Location(ctx);
        operands=IR.get_result.([op_3, op_4]),
        results=[],
        attributes=[
            IR.NamedAttribute(ctx, "odsOperandSegmentSizes", IR.Attribute(API.mlirDenseI32ArrayGet(ctx, 3, Int32[1, 1, 0])))])
    
    rewrite_block = IR.Block()
    push!(rewrite_block, op_4)
    push!(rewrite_block, replace_op)
    
    rewrite_region = IR.Region()
    push!(rewrite_region, rewrite_block)
    
    rewrite_op = IR.create_operation(
        "pdl.rewrite",
        IR.Location(ctx);
        operands=[IR.get_result(op_3)],
        results=[],
        attributes=[
            IR.NamedAttribute(ctx, "odsOperandSegmentSizes", IR.Attribute(API.mlirDenseI32ArrayGet(ctx, 2, Int32[1, 0])))],
        owned_regions=[rewrite_region])
    
    pattern_block = IR.Block()
    push!(pattern_block, op_0)
    push!(pattern_block, op_1)
    push!(pattern_block, op_2)
    push!(pattern_block, op_3)
    push!(pattern_block, rewrite_op)
    
    pattern_region = IR.Region()
    push!(pattern_region, pattern_block)
    
    pattern_op = IR.create_operation(
        "pdl.pattern",
        IR.Location(ctx);
        operands=[],
        results=[],
        attributes=[
            IR.NamedAttribute(ctx, "benefit", IR.Attribute(ctx, Int16(1)))],
        owned_regions=[pattern_region])
end

# @show manual_pattern_op

manual_pattern_module = IR.MModule(ctx, IR.Location(ctx))
push!(IR.get_body(manual_pattern_module), manual_pattern_op)

@show IR.get_operation(manual_pattern_module)

pdl_pattern = API.beaverPDLPatternGet(manual_pattern_module)
pattern_set = API.beaverRewritePatternSetGet(ctx)
pattern_set = API.beaverPatternSetAddOwnedPDLPattern(pattern_set, pdl_pattern)

using MLIR.Brutus
f(a, b) = a+b
source_op = Brutus.@code_mlir f(2, 3)

API.beaverApplyOwnedPatternSetOnOperation(source_op, pattern_set)

@show source_op
