* How do we measure the **similarity**?

  1. bag of words (word counts); we could this approach to represent an article

     1. Then we would **multiply** this word counts to another document's word counts, the more the number will be, the more similar they are.
     2. She said we might need to **normalize** the frequency of each word; because otherwise words with more frequency will dominate **rare words.**
     3. What are **rare words** and ,in contrast, **common words**?
        1. Common words: are words that appear commonly in a single document (**common locally**). In contrast, here, we want to down weight these words.
        2. While **rare words**, appear rarely in the corpus (**rare globally**). So, we're looking for a way to up weight these words.

     * seems like we need to find a trade-off between rare-words and common words.

  2. One way to address this problem is `TF-IDF` 

     1. `TF-IDF` stands for `Term Frequency - Inverse Document Frequency`
     2. Term frequency: count the appearance of each word in the document, no the corpus.
     3. Inverse Document frequency: which is also a matrix, it is calculated by below formula:

     ```markdown
     log #docs/(1+#docs using that word)
     	max: we have a word that exists in all documents
     	log 1 = 0
     	min: we have a word that appears scarcely in all documents
     	log large/number => a large number 
     ```

     * So the `TF_IDF` approach **puts more values on rare words**; while decreasing the value of common words across the corpus.

* How do we **search over articles**?

  1. One approach was using `KNN` by different distant measurement.
  2. Another approach is clustering (`unsupervised learning`).
