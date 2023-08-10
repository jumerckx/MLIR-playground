module Utils
export @code_ircode, @dot

using InteractiveUtils

code_ircode = Base.code_ircode

macro code_ircode(ex0...)
    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(@__MODULE__, :code_ircode, ex0)
    quote
        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end

function create_dotgraph(ir::Core.Compiler.IRCode; clip=false)
    blocks = ir.cfg.blocks
    edges = []
    labels = []
    for (i, block) in enumerate(blocks)
        r = block.stmts.start:block.stmts.stop
        instructions = ir.stmts.inst[r]
        # show instructions in label, left-aligned 
        push!(labels, "$i [label=\"$i\\n$(join(instructions, "\\l"))\\l\"]")
        push!(edges, string.(block.preds) .* " -> $i"...)
    end
    graph = """
    digraph G {
    node [margin=1 fontname="consolas" fontsize=32 width=1 shape=box style=filled]
    $(join(labels, "\n"))
    $(join(edges, "\n"))
    }
    """
    clip && clipboard(graph)
    return graph
end

macro dot(ex)
    ir, ret = (__module__).eval(:(@code_ircode $(ex)))
    command = create_dotgraph(ir, clip=false)
    return command
end

module DebugIR
using MLIR.IR

function into(operation::IR.Operation)
    out = []
    for region in IR.RegionIterator(operation)
        push!(out, region)
    end
    return out
end
function into(region::IR.Region)
    out = []
    for block in IR.BlockIterator(region)
        push!(out, block)
    end
    return out
end
function into(block::IR.Block)
    out = []
    for operation in IR.OperationIterator(block)
        push!(out, operation)
    end
    return out
end

struct DOperation
    regions::Union{Vector,Nothing}
end
struct DRegion
    blocks::Vector
end
struct DBlock
    operations::Vector
end
f(element::IR.Operation) = begin
    regions = into(element)
    if length(regions) == 0
        return DOperation(nothing)
    end
    return DOperation(f.(into(element)))
end
f(element::IR.Region) = DRegion(f.(into(element)))
f(element::IR.Block) = DBlock(f.(into(element)))

end
end