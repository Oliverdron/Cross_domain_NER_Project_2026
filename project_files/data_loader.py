from datasets import load_dataset
import os
import shutil


OUT_DIR = "data_iob2"
os.makedirs(OUT_DIR, exist_ok=True)


def write_hf_iob2(dataset_split, output_path, token_field="tokens", tag_field="ner_tags"):
    """
    Write a Hugging Face NER split to:
    token<TAB>tag

    Works for both:
    - integer labels with ClassLabel
    - string labels already stored directly
    """
    tag_feature = dataset_split.features[tag_field]

    # Case 1: labels are integers with a ClassLabel mapping
    has_class_names = hasattr(tag_feature, "feature") and hasattr(tag_feature.feature, "names")

    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            tokens = example[token_field]
            raw_tags = example[tag_field]

            if has_class_names:
                label_names = tag_feature.feature.names
                tags = [label_names[tag_id] for tag_id in raw_tags]
            else:
                tags = raw_tags  # already strings

            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


def copy_universal_file(src_path, dst_path):
    shutil.copyfile(src_path, dst_path)


# ----------------------------
# 1. UNIVERSAL
# ----------------------------
copy_universal_file("en_ewt-ud-train.iob2", f"{OUT_DIR}/universal_train.iob2")
copy_universal_file("en_ewt-ud-dev.iob2", f"{OUT_DIR}/universal_dev.iob2")
copy_universal_file("en_ewt-ud-test-masked.iob2", f"{OUT_DIR}/universal_test_masked.iob2")

print("Universal files copied.")


# ----------------------------
# 2. NEWS
# ----------------------------
news = load_dataset("conll2003", trust_remote_code=True)

write_hf_iob2(news["train"], f"{OUT_DIR}/news_train.iob2")
write_hf_iob2(news["validation"], f"{OUT_DIR}/news_dev.iob2")
write_hf_iob2(news["test"], f"{OUT_DIR}/news_test.iob2")

print("News files written.")


# ----------------------------
# 3. ASTRO
# ----------------------------
astro = load_dataset("adsabs/WIESP2022-NER")

# Optional: inspect feature names if needed
print("Astro features:", astro["train"].features)

write_hf_iob2(astro["train"], f"{OUT_DIR}/astro_train.iob2")
write_hf_iob2(astro["validation"], f"{OUT_DIR}/astro_dev.iob2")
write_hf_iob2(astro["test"], f"{OUT_DIR}/astro_test.iob2")

print("Astro files written.")
print("Done.")