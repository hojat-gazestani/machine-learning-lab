from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    #for list_idx, docs in enumerate(results, start=1):
    #    print(f"\n====== Retrieval list {list_idx} =====")

    #    for rank, doc in enumerate(docs, start=1):
    #        print(f"Rank {rank}")
    #        print(doc.page_content[:100].replace("\n", " "))
    #        print("-" * 60)
    ## Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    ## Iterate through each list of ranked documents
    #for docs in results:
    for list_idx, docs in enumerate(results, start=1):

    #    print(f"\nProccessing list {list_idx}")

    #    # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
    #        score = 1 / (rank + k)
    #        print(
    #                f"List: {list_idx}",
    #                f"Rank: {rank}",
    #                f"Add: {score:.5f}"
    #        )
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)


    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse = True)
    ]

    #print(f"\n====================== FINAL RANKING ====================")
    #for i, (doc, score) in enumerate(reranked_results, start=1):
    #    print(
    #            f"Final Rank {i} |"
    #            f"Score {score:.5f}"
    #    )
    #    print(doc.page_content[:100].replace("\n", " "))
    #    print("-" * 59)
    ## Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
