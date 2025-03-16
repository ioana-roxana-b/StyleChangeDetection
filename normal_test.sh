python3 main.py \
      --problem normal \
      --text-name HVIII \
      --input-text-path Corpus/Shakespeare/HVIII.txt \
      --generate-features False \
      --features-path Outputs/Q0/HVIII/chapter_features.csv \
      --classifier-config-path classification_configs/classifiers_config.json \
      --classifier-config-key all \
      --label 'DOS|TOL|Shakespeare|Fletcher'