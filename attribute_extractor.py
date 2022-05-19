# Load dependencies
import collections
import numpy as np

import tensorflow as tf
from transformers import DefaultDataCollator, TFAutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset

class AttributeExtractor():

    def __init__(self, model_ckpt: str) -> None:
        self.model_ckpt = model_ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(self.model_ckpt)
        self.negative = True
        self.max_length = 384
        self.doc_stride = 128
        self.min_null_score = 0
        return
    
    # Prepare tokenized example features
    def __prepare_example_features(self, examples: dict) -> dict:
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        pad_on_right = self.tokenizer.padding_side == "right"
        max_length = 384  # The maximum length of a feature (question and context)
        doc_stride = 128  # The allowed overap between two part of the context when splitting is performed.
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    # Extract features from processed examples
    def extract_features(self, examples: dict) -> None:
        data_collator = DefaultDataCollator(return_tensors="tf")
        dataset = Dataset.from_dict(examples)
        features = dataset.map(
            self.__prepare_example_features,
            batched=True,
            remove_columns=dataset.column_names)
        tf_features = features.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            shuffle=False,
            batch_size=32,
            collate_fn=data_collator)
        return dataset, features, tf_features

    # Function that postprocessed predictions by model
    def postprocess_qa_predictions(self,
        examples,
        features,
        all_start_logits,
        all_end_logits,
        n_best_size=20,
        max_answer_length=30,
    ):
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            valid_answers = []

            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(
                    self.tokenizer.cls_token_id
                )
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if self.min_null_score is None or self.min_null_score < feature_null_score:
                    self.min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        try:
                            start_char = offset_mapping[start_index][0]
                            end_char = offset_mapping[end_index][1]
                        except:
                            continue
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char:end_char],
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                    0
                ]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not self.negative:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = (
                    best_answer["text"] if best_answer["score"] > self.min_null_score else ""
                )
                predictions[example["id"]] = answer

        return predictions
    
    def predict(self, examples: dict, negative=True, min_null_score=5) -> dict:
        self.min_null_score = 5
        self.negative = negative
        dataset, features, tf_features = self.extract_features(examples)
        raw_predictions = self.model.predict(tf_features)
        return self.postprocess_qa_predictions(
            dataset,
            features, 
            raw_predictions['start_logits'],
            raw_predictions['end_logits']
        )