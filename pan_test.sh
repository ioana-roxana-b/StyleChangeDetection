python3 main.py \
      --train-dataset-path Corpus_pan/pan14/pan14-authorship-verification-train-corpus-2014-04-22/pan14-authorship-verification-training-corpus-spanish-articles-2014-04-22 \
      --test-dataset-path Corpus_pan/pan14/pan14-authorship-verification-test-corpus2-2014-04-22/pan14-authorship-verification-test-corpus2-spanish-articles-2014-04-22 \
      --train-truth-path Corpus_pan/pan14/pan14-authorship-verification-train-corpus-2014-04-22/pan14-authorship-verification-training-corpus-spanish-articles-2014-04-22/truth.txt \
      --test-truth-path Corpus_pan/pan14/pan14-authorship-verification-test-corpus2-2014-04-22/pan14-authorship-verification-test-corpus2-spanish-articles-2014-04-22/truth.txt \
      --generate-features True \
      --features-path-train Corpus_pan/pan/train-14-spanish-articles-corpus2 \
      --features-path-test Corpus_pan/pan/test-14-spanish-articles-corpus2 \
      --classifier-config-path classification_configs/pan_classifier_config.json \
      --classifier-config-key all \
      --language es \
      --wan-config C1