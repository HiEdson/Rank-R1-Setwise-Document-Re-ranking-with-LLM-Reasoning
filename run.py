import logging
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from rank_r1_core import SearchResult, RankR1SetwiseLlmRanker, SetwiseLlmRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random

random.seed(929)
logger = logging.getLogger(__name__)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):
    # Initialization
    # Use reasoning if explicitly requested or if a prompt file is provided
    use_reasoning = args.setwise.reasoning or (args.run.prompt_file is not None)
    
    if use_reasoning:
        logger.info("Initializing SetwiseLlmRanker with R1 Reasoning enabled")
    else:
        logger.info("Initializing standard SetwiseLlmRanker")
        
    ranker = SetwiseLlmRanker(
        model_name_or_path=args.run.model_name_or_path,
        tokenizer_name_or_path=args.run.tokenizer_name_or_path,
        device=args.run.device,
        cache_dir=args.run.cache_dir,
        num_child=args.setwise.num_child,
        scoring=args.run.scoring,
        method=args.setwise.method,
        num_permutation=args.setwise.num_permutation,
        k=args.setwise.k,
        reasoning=use_reasoning
    )

    # Data Loading (Pyserini, ir_datasets, or Local files)
    query_map = {}
    docstore = None
    local_docs = {}

    if args.run.queries_path is not None:
        logger.info(f"Loading local queries from {args.run.queries_path}")
        with open(args.run.queries_path, 'r') as f:
            queries_data = json.load(f)
            for qid, text in queries_data.items():
                query_map[str(qid)] = ranker.truncate(text, args.run.query_length)

    if args.run.docs_path is not None:
        logger.info(f"Loading local docs from {args.run.docs_path}")
        with open(args.run.docs_path, 'r') as f:
            local_docs = json.load(f)

    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        for query in dataset.queries_iter():
            if query.query_id not in query_map:
                query_map[query.query_id] = ranker.truncate(query.text, args.run.query_length)
        docstore = dataset.docs_store()
    elif args.run.pyserini_index is not None:
        topics = get_topics(args.run.pyserini_index + '-test')
        for topic_id in list(topics.keys()):
            if str(topic_id) not in query_map:
                query_map[str(topic_id)] = ranker.truncate(topics[topic_id]['title'], args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index + '.flat')

    # Load first stage rankings
    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map.get(current_qid, ""), current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.run.hits:
                continue
            
            # Fetch text
            if docid in local_docs:
                data = local_docs[docid]
                text = f"{data.get('title', '')} {data['text']}".strip()
            elif args.run.ir_dataset_name is not None:
                doc = docstore.get(docid)
                text = f"{getattr(doc, 'title', '')} {doc.text}".strip()
            elif docstore is not None:
                data = json.loads(docstore.doc(docid).raw())
                text = f"{data.get('title', '')} {data['text']}".strip()
            else:
                logger.warning(f"DocID {docid} not found in any data source.")
                text = ""
            
            text = ranker.truncate(text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map.get(current_qid, ""), current_ranking[:args.run.hits]))

    # Execution Loop
    reranked_results = []
    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if not query or not ranking: continue
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))
    toc = time.time()

    print(f'Avg time per query: {(toc-tic)/len(reranked_results)}')
    write_run_file(args.run.save_path, reranked_results, 'RankR1Core')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, required=True, help='Path to TREC run file.')
    run_parser.add_argument('--save_path', type=str, required=True, help='Path to save results.')
    run_parser.add_argument('--model_name_or_path', type=str, required=True)
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None)
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--queries_path', type=str, default=None, help='Path to local queries JSON file.')
    run_parser.add_argument('--docs_path', type=str, default=None, help='Path to local docs JSON file.')
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--lora_path_or_name', type=str, default=None, help='Optional LoRA adapter.')
    run_parser.add_argument('--prompt_file', type=str, default=None, help='Required for Rank-R1.')
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=19)
    setwise_parser.add_argument('--method', type=str, default='heapsort', choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)
    setwise_parser.add_argument('--reasoning', action='store_true', help='Enable R1-style reasoning (hardcoded prompt).')

    args = parse_args(parser, commands)
    if not args.run or not args.setwise:
        print("Usage: python run.py run [run_args] setwise [setwise_args]")
        sys.exit(1)
    
    main(args)




# python run.py run --model_name_or_path google/flan-t5-small 
# --run_path sample_data/bm25_simulated.run --save_path sample_data/reranked.txt 
# --queries_path sample_data/queries.json --docs_path sample_data/docs.json 
# --device cpu setwise --num_child 2 --k 5 --reasoning