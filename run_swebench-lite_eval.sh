CHUNKING_STRATEGY=ast
EMBEDDING_MODEL=BGE-base
GENERATION_MODEL=gemini-2.5-pro

OUTPUT_JSONL=swebench/${CHUNKING_STRATEGY}_chunking/swebench-lite_${CHUNKING_STRATEGY}-chunking_${EMBEDDING_MODEL}_${GENERATION_MODEL}_generations_swebench-lite.jsonl

./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $OUTPUT_JSONL "" princeton-nlp/SWE-bench_Lite test
