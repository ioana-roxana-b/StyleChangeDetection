python3 main.py \
      --problem dialogism \
      --text-name WinnieThePooh \
      --input-text-path Corpus/project-dialogism-novel-corpus-master/data/WinnieThePooh/quotation_info.csv \
      --generate-features False \
      --features-path Outputs/Q2/WinnieThePooh/sentence_tf_idf.csv \
      --classifier-config-path classification_configs/WinnieThePooh_classifiers_config.json \
      --classifier-config-key all