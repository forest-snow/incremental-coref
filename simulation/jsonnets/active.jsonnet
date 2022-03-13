local preco = import "preco.jsonnet";
local base = import "base.jsonnet";
local data = import "data.jsonnet";
local encoders = import "encoders.jsonnet";
local qbcoref = import "qbcoref.jsonnet";
{
    local Active(seed, spans, docs, strategy, cycles) = (
        {
            num_spans: spans,
            max_docs: docs,
            seed: seed,
            strategy: strategy,
            num_cycles: cycles,
            save_ckpt: false,
            start: 1,
        }

    ),

    local ActiveDebug() = (
        base.base +
        encoders.spanbert_large +
        encoders.finetune_top(24) +
        data.Preco +
        base.Name("active_debug") +
        Active(67, 10, 5, "random", 3) +
        {singleton_eval: true,
            mentions: true,
            num_train_examples: 50,
            num_epochs: 2
        }
    ),

    local ActivePreco(seed, spans, docs, strategy, cycles) = (
        base.base +
        encoders.spanbert_large +
        encoders.finetune_top(24) +
        data.Preco +
        base.Name("preco") +
        Active(seed, spans, docs, strategy, cycles) +
        {singleton_eval: true,
            mentions: true,
            num_train_examples: 2500,
        }
    ),

    local UserStudy(session, spans, docs) = (
        ActivePreco(67, spans, docs, "ment-ent", 1) +
        base.Name("userstudy") +
        {user: "",
        session: session
        }
    ),

    local ActiveQb(seed, spans, docs, strategy, cycles) = (
        base.base +
        encoders.spanbert_large +
        encoders.finetune_top(24) +
        data.Qbcoref_split(0) +
        base.Name("qbcoref") +
        Active(seed, spans, docs, strategy, cycles) +
        {singleton_eval: true,
         mentions: true,
         num_train_examples: 240,
        }
    ),

    active_debug: {"active_debug": ActiveDebug()},

    active_preco: {
        ["active_preco_"+ seed + "_" +  num_spans + "_" + max_docs + "_" + strategy + "_" + cycles]: ActivePreco(seed, num_spans, max_docs, strategy, cycles)
        for num_spans in [0, 20, 50]
        for max_docs in [0, 1, 5, 20, 50]
        for seed in [67, 312, 57, 29, 8]
        for strategy in ["random", "ment-ent", "clust-ent", "cond-ent", "joint-ent", "random-ment", "li-ent"]
        for cycles in [0, 6, 15]
    },

    userstudy: {
        ["userstudy_"+session+"_"+num_spans+"_"+max_docs]: UserStudy(session, num_spans, max_docs)
        for session in [1, 2]
        for num_spans in [100, 200]
        for max_docs in [10, 100]
    },

    active_qbcoref: {
        ["active_qbcoref_"+ seed + "_" +  num_spans + "_" + max_docs + "_" + strategy + "_" + cycles]: ActiveQb(seed, num_spans, max_docs, strategy, cycles)
        for num_spans in [0, 20, 40]
        for max_docs in [0, 1, 5, 20, 40]
        for seed in [67, 312, 57, 29, 8]
        for strategy in ["random", "ment-ent", "clust-ent", "cond-ent", "joint-ent", "random-ment", "li-ent"]
        for cycles in [0, 10, 20]
    },



}
