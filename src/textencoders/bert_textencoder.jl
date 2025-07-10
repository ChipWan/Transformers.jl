# 导入必要的模块和函数
using ..WordPieceModel
using ..WordPieceModel: DAT
using FuncPipelines
using TextEncodeBase
using TextEncodeBase: nested2batch, nestedcall
using TextEncodeBase: BaseTokenization, WrappedTokenization, MatchTokenization, Splittable,
    ParentStages, TokenStages, SentenceStage, WordStage, Batch, Sentence, getvalue, getmeta
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm

# ============================================================================
# BERT 分词器结构定义
# ============================================================================

"""
    BertCasedPreTokenization <: BaseTokenization

BERT 大小写敏感的预分词器。
保留原始文本的大小写信息，适用于需要区分大小写的任务。
"""
struct BertCasedPreTokenization <: BaseTokenization end

"""
    BertUnCasedPreTokenization <: BaseTokenization

BERT 大小写不敏感的预分词器。
将所有文本转换为小写，适用于大多数NLP任务。
"""
struct BertUnCasedPreTokenization <: BaseTokenization end

# 定义分词行为
TextEncodeBase.splitting(::BertCasedPreTokenization, s::SentenceStage) = 
    bert_cased_tokenizer(getvalue(s))

TextEncodeBase.splitting(::BertUnCasedPreTokenization, s::SentenceStage) = 
    bert_uncased_tokenizer(getvalue(s))

# 类型别名，用于统一处理两种BERT分词器
const BertTokenization = Union{BertCasedPreTokenization, BertUnCasedPreTokenization}

# 自定义显示方法
Base.show(io::IO, ::BertCasedPreTokenization) = print(io, nameof(bert_cased_tokenizer))
Base.show(io::IO, ::BertUnCasedPreTokenization) = print(io, nameof(bert_uncased_tokenizer))

# ============================================================================
# BERT 文本编码器构造函数
# ============================================================================

"""
    BertTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)

BERT文本编码器的核心构造函数。

# 参数
- `tkr`: 分词器实例
- `vocab`: 词汇表，包含所有可能的token
- `process`: 文本处理管道
- `startsym`: 序列开始符号，通常为"[CLS]"
- `endsym`: 序列结束符号，通常为"[SEP]"  
- `padsym`: 填充符号，通常为"[PAD]"
- `trunc`: 截断长度限制

# 数学表示
对于输入序列 x = [x₁, x₂, ..., xₙ]，编码后的序列为：
encoded = [startsym, x₁, x₂, ..., xₙ, endsym]
"""
function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary{String}, process,
                         startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int})
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

# 函数分发：根据分词器类型自动选择对应的预分词器
BertTextEncoder(::typeof(bert_cased_tokenizer), args...; kws...) =
    BertTextEncoder(BertCasedPreTokenization(), args...; kws...)

BertTextEncoder(::typeof(bert_uncased_tokenizer), args...; kws...) =
    BertTextEncoder(BertUnCasedPreTokenization(), args...; kws...)

# WordPiece分词器的特殊处理
BertTextEncoder(bt::BertTokenization, wordpiece::WordPiece, args...; kws...) =
    BertTextEncoder(WordPieceTokenization(bt, wordpiece), args...; kws...)

"""
处理WordPiece分词器，支持匹配特定token的功能。

# 数学原理
WordPiece算法通过最大化语言模型的似然概率来选择最佳的子词分割：
P(sentence) = ∏ᵢ P(tokenᵢ | context)
其中每个token可能是完整单词或子词片段。
"""
function BertTextEncoder(t::WordPieceTokenization, args...; match_tokens = nothing, kws...)
    if isnothing(match_tokens)
        return BertTextEncoder(TextTokenizer(t), Vocab(t.wordpiece), args...; kws...)
    else
        # 确保match_tokens是数组格式
        match_tokens = match_tokens isa AbstractVector ? match_tokens : [match_tokens]
        return BertTextEncoder(
            TextTokenizer(MatchTokenization(t, match_tokens)), 
            Vocab(t.wordpiece), 
            args...; 
            kws...
        )
    end
end

"""
通用分词器处理函数，支持匹配特定token。
"""
function BertTextEncoder(t::AbstractTokenization, vocab::AbstractVocabulary, args...; 
                        match_tokens = nothing, kws...)
    if isnothing(match_tokens)
        return BertTextEncoder(TextTokenizer(t), vocab, args...; kws...)
    else
        match_tokens = match_tokens isa AbstractVector ? match_tokens : [match_tokens]
        return BertTextEncoder(
            TextTokenizer(MatchTokenization(t, match_tokens)), 
            vocab, 
            args...; 
            kws...
        )
    end
end

# ============================================================================
# 词汇表处理函数
# ============================================================================

"""
    _wp_vocab(wp::WordPiece) -> Vector{String}

从WordPiece模型中提取词汇表。

# 算法说明
遍历WordPiece的trie结构，按照索引顺序重建词汇表：
vocab[index[id]] = string_representation
"""
function _wp_vocab(wp::WordPiece)
    vocab = Vector{String}(undef, length(wp.trie))
    for (str, id) in wp.trie
        vocab[wp.index[id]] = str
    end
    return vocab
end

# 创建词汇表实例
TextEncodeBase.Vocab(wp::WordPiece) = Vocab(_wp_vocab(wp), DAT.decode(wp.trie, wp.unki))

# ============================================================================
# 带默认参数的构造函数
# ============================================================================

"""
    BertTextEncoder(tkr, vocab, process; startsym="[CLS]", endsym="[SEP]", padsym="[PAD]", trunc=nothing)

带有默认BERT特殊符号的构造函数。

# BERT特殊符号说明
- `[CLS]`: 分类标记，用于分类任务的句子表示
- `[SEP]`: 分隔符，用于分离不同的句子
- `[PAD]`: 填充符，用于对齐不同长度的序列
"""
function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process;
                         startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]", trunc = nothing)
    # 检查特殊符号是否在词汇表中
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    
    return BertTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

"""
    BertTextEncoder(tkr, vocab; fixedsize=false, trunc_end=:tail, pad_end=:tail, process=nothing, ...)

完整的BERT文本编码器构造函数，包含所有预处理选项。

# 参数说明
- `fixedsize`: 是否固定输出序列长度
- `trunc_end`: 截断位置（:head 或 :tail）
- `pad_end`: 填充位置（:head 或 :tail）
- `process`: 自定义处理管道
"""
function BertTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary;
                         fixedsize = false, trunc_end = :tail, pad_end = :tail, process = nothing,
                         kws...)
    # 创建基础编码器
    enc = BertTextEncoder(tkr, vocab, TextEncodeBase.process(AbstractTextEncoder); kws...)
    
    # 应用默认的BERT预处理管道
    return BertTextEncoder(enc) do e
        bert_default_preprocess(; 
            trunc = e.trunc, 
            startsym = e.startsym, 
            endsym = e.endsym, 
            padsym = e.padsym,
            fixedsize, 
            trunc_end, 
            pad_end, 
            process
        )
    end
end

# 构建器模式支持
BertTextEncoder(builder, e::TrfTextEncoder) = TrfTextEncoder(builder, e)

# ============================================================================
# BERT 默认预处理管道
# ============================================================================

"""
    bert_default_preprocess(; kwargs...)

BERT模型的默认预处理管道。

# 数学表示
对于输入序列 x = [x₁, x₂, ..., xₙ]，处理步骤如下：

1. 添加特殊符号：
   sequence = [CLS] + x + [SEP]

2. 对于双句子输入：
   sequence = [CLS] + sentence₁ + [SEP] + sentence₂ + [SEP]

3. 生成segment_ids：
   - sentence₁的所有token（包括[CLS]和第一个[SEP]）：segment_id = 0
   - sentence₂的所有token（包括第二个[SEP]）：segment_id = 1

4. 生成attention_mask：
   - 实际token位置：mask = 1
   - 填充位置：mask = 0

5. 截断和填充：
   if len(sequence) > max_length:
       sequence = truncate(sequence, max_length, trunc_end)
   if len(sequence) < max_length:
       sequence = pad(sequence, max_length, pad_end, padsym)
"""
function bert_default_preprocess(; startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]",
                                 fixedsize = false, trunc = nothing, trunc_end = :tail, pad_end = :tail,
                                 process = nothing)
    # 获取截断和填充函数
    truncf = get_trunc_pad_func(fixedsize, trunc, trunc_end, pad_end)
    maskf = get_mask_func(trunc, pad_end)
    
    # 如果没有提供自定义处理管道，使用默认管道
    if isnothing(process)
        process =
            # 步骤1: 对输入进行分组处理
            Pipeline{:token}(grouping_sentence, :token) |>
            # 步骤2: 添加特殊符号并计算segment信息
            # 使用序列模板：[CLS] + input₁ + [SEP] + input₂ + [SEP] + ...
            Pipeline{:token_segment}(
                SequenceTemplate(
                    ConstTerm(startsym, 1),                    # [CLS] with segment_id=1
                    InputTerm{String}(1),                      # 第一个输入序列 with segment_id=1
                    ConstTerm(endsym, 1),                      # [SEP] with segment_id=1
                    RepeatedTerm(                              # 可重复的第二个序列部分
                        InputTerm{String}(2),                  # 第二个输入序列 with segment_id=2
                        ConstTerm(endsym, 2);                  # [SEP] with segment_id=2
                        dynamic_type_id = true                 # 动态类型ID
                    )
                ), :token
            ) |>
            # 步骤3: 提取token序列
            Pipeline{:token}(nestedcall(first), :token_segment) |>
            # 步骤4: 提取segment序列
            Pipeline{:segment}(nestedcall(last), :token_segment)
    end
    
    # 构建完整的处理管道
    return Pipeline{:token}(nestedcall(string_getvalue), 1) |>     # 转换为字符串
        process |>                                                  # 应用处理管道
        Pipeline{:attention_mask}(maskf, :token) |>                # 生成注意力掩码
        Pipeline{:token}(truncf(padsym), :token) |>                # 截断和填充token
        Pipeline{:token}(nested2batch, :token) |>                  # 转换为批次格式
        Pipeline{:segment}(truncf(1), :segment) |>                 # 截断和填充segment
        Pipeline{:segment}(nested2batch, :segment) |>              # 转换segment为批次格式
        Pipeline{:sequence_mask}(identity, :attention_mask) |>     # 生成序列掩码
        PipeGet{(:token, :segment, :attention_mask, :sequence_mask)}()  # 返回所需的输出
end
