import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    #init the dict with the probabilities
    prob_values = dict.fromkeys(corpus.keys(), 0)
    # links of the page input 
    links = corpus[page]
    # if the current page has links
    if links:
        # assign for every link in the corpus
        for p in corpus: 
            prob_values[p] = (1-damping_factor)/len(corpus)
            # assign for the links to the page
            if p in links:
                prob_values[p] += damping_factor/len(links)
    else:
        for p in corpus:
            prob_values[p] = 1/len(corpus)

    return prob_values


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # keep counting the pages {page1: 0, page2: 0,---}
    count_page = dict.fromkeys(corpus.keys(), 0)
    # current page at random 
    page = random.choice(list(corpus.keys()))

    for i in range(n):
        # count the page selected 
        count_page[page] += 1
        model_to_sample = transition_model(corpus, page, damping_factor)
        # update the page
        page = random.choices(
        list(model_to_sample.keys()),
        weights=list(model_to_sample.values())
        )[0]

    pageRank = {page: count/n for page, count in count_page.items()}
    
    return pageRank 


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {page: 1/N for page in corpus}  # initial assumption 1/N
    while True:
        new_prob = {}
        for p in corpus:
            # init the probability associate for those pages with links
            prob_links = 0
            # include all pages i than link to p 
            for i in corpus:
                # calculate the probability distribution over which page to visit next, giving the current i page
                model_i = transition_model(corpus, i, damping_factor)
                # calculate the probability
                prob_links += pagerank[i]*model_i[p]

            new_prob[p] = prob_links
        #check the convergence
        diff = sum(abs(new_prob[p] - pagerank[p]) for p in corpus)
        if diff < 1e-3:
            break

        pagerank = new_prob

    total = sum(pagerank.values())
    pagerank = {p: v / total for p, v in pagerank.items()}
    return pagerank


if __name__ == "__main__":
    main()
