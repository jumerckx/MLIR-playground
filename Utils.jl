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
end