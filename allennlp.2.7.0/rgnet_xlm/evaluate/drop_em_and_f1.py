from typing import Tuple, List, Union, Dict

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from rgnet_xlm.evaluate.drop_eval import (get_metrics as drop_em_and_f1,
                       answer_json_to_strings)

class DropEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._supporting_exact = 0.0
        self._supporting_total = 0.0
        self._judge_count = 0
        self._comp_count = 0
        self._open_count = 0
        self._unsure_count = 0
        self._extract_count = 0
        self._arithmetic_count = 0
        self._arithmetic_open_count = 0
        self._judge_em = 0.0
        self._comp_em = 0.0
        self._open_em = 0.0
        self._unsure_em = 0.0
        self._extract_em = 0.0
        self._arithmetic_em = 0.0
        self._arithmetic_open_em = 0.0
        self._judge_f1 = 0.0
        self._comp_f1 = 0.0
        self._open_f1 = 0.0
        self._unsure_f1 = 0.0
        self._extract_f1 = 0.0
        self._arithmetic_f1 = 0.0
        self._arithmetic_open_f1 = 0.0

    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths, history_answer,answer_type,node_bag):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth, history_answer, answer_type, node_bag)
            scores_for_ground_truths.append(score)

        return max(scores_for_ground_truths)

    @overrides
    def __call__(self, prediction: Union[str, List], ground_truths: List, history_answer: Dict,answer_type: List,node_bag:Dict):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].
        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
        #print("ground_truth_answer_strings",ground_truth_answer_strings)
        exact_match, f1_score, supporting_exact, supporting_count, type_EM, type_f1, type_count = self.metric_max_over_ground_truths(
            drop_em_and_f1,
            prediction,
            ground_truth_answer_strings,
            history_answer,
            answer_type,
            node_bag
        )
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1
        self._supporting_exact += supporting_exact
        self._supporting_total += supporting_count
        self._judge_count += 0
        self._comp_count += 0
        self._open_count += 0
        self._unsure_count += 0
        self._extract_count += 0
        self._arithmetic_count += 0
        self._arithmetic_open_count += 0
        self._judge_em += 0.0
        self._comp_em += 0.0
        self._open_em += 0.0
        self._unsure_em += 0.0
        self._extract_em += 0.0
        self._arithmetic_em += 0.0
        self._arithmetic_open_em += 0.0
        self._judge_f1 += 0.0
        self._comp_f1 += 0.0
        self._open_f1 += 0.0
        self._unsure_f1 += 0.0
        self._extract_f1 += 0.0
        self._arithmetic_f1 += 0.0
        self._arithmetic_open_f1 += 0.0
        for type_e, type_E in type_EM.items():
            if type_E == 1.0:
                if type_e == '1':
                    self._judge_em += 1.0
                elif type_e == '2':
                    self._comp_em += 1.0
                elif type_e == '4':
                    self._open_em += 1.0
                elif type_e == '3':
                    self._arithmetic_em += 1.0
                elif type_e == '5':
                    self._unsure_em += 1.0
                elif type_e == '6':
                    self._extract_em += 1.0
                elif type_e == '7':
                    self._arithmetic_open_em += 1.0
        for type_f, type_F in type_f1.items():
            if type_F > 0.0:
                if type_f == '1':
                    self._judge_f1 += type_F
                elif type_f == '2':
                    self._comp_f1 += type_F
                elif type_f == '3':
                    self._arithmetic_f1 += type_F
                elif type_f == '4':
                    self._open_f1 += type_F
                elif type_f == '5':
                    self._unsure_f1 += type_F
                elif type_f == '6':
                    self._extract_f1 += type_F
                elif type_f == '7':
                    self._arithmetic_open_f1 += type_F
        for type_c, type_C in type_count.items():
            if type_C == 1:
                if type_c == '1':
                    self._judge_count += 1
                elif type_c == '2':
                    self._comp_count += 1
                elif type_c == '3':
                    self._arithmetic_count += 1
                elif type_c == '4':
                    self._open_count += 1
                elif type_c == '5':
                    self._unsure_count += 1
                elif type_c == '6':
                    self._extract_count += 1
                elif type_c == '7':
                    self._arithmetic_open_count += 1


    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[
        Union[float, int], Union[float, int], Union[float, int], float, Union[float, int], Union[float, int], Union[
            float, int], Union[float, int], Union[float, int], Union[float, int], Union[float, int], Union[float, int],
        Union[float, int], Union[float, int], Union[float, int], Union[float, int], Union[float, int], Union[
            float, int]]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        supporting_ratio = self._supporting_exact / self._supporting_total if self._supporting_total > 0 else 0
        supporting_right = self._supporting_exact
        judge_em = self._judge_em / self._judge_count if self._judge_count > 0 else 0
        comp_em = self._comp_em / self._comp_count if self._comp_count > 0 else 0
        open_em = self._open_em / self._open_count if self._open_count > 0 else 0
        unsure_em = self._unsure_em / self._unsure_count if self._unsure_count > 0 else 0
        extract_em = self._extract_em / self._extract_count if self._extract_count > 0 else 0
        arithmetic_em = self._arithmetic_em / self._arithmetic_count if self._arithmetic_count > 0 else 0
        arithmetic_open_em = self._arithmetic_open_em / self._arithmetic_open_count if self._arithmetic_open_count > 0 else 0
        judge_f1 = self._judge_f1 / self._judge_count if self._judge_count > 0 else 0
        comp_f1 = self._comp_f1 / self._comp_count if self._comp_count > 0 else 0
        open_f1 = self._open_f1 / self._open_count if self._open_count > 0 else 0
        unsure_f1 = self._unsure_f1 / self._unsure_count if self._unsure_count > 0 else 0
        extract_f1 = self._extract_f1 / self._extract_count if self._extract_count > 0 else 0
        arithmetic_f1 = self._arithmetic_f1 / self._arithmetic_count if self._arithmetic_count > 0 else 0
        arithmetic_open_f1 = self._arithmetic_open_f1 / self._arithmetic_open_count if self._arithmetic_open_count > 0 else 0

        if reset:
            self.reset()
        return exact_match, f1_score, supporting_ratio, supporting_right, judge_em, \
               comp_em, open_em, unsure_em, extract_em, arithmetic_em, arithmetic_open_em, \
               judge_f1, comp_f1, open_f1, unsure_f1, extract_f1, arithmetic_f1, arithmetic_open_f1

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._supporting_exact = 0
        self._supporting_total = 0
        self._judge_count = 0
        self._comp_count = 0
        self._open_count = 0
        self._unsure_count = 0
        self._extract_count = 0
        self._arithmetic_count = 0
        self._arithmetic_open_count = 0
        self._judge_em = 0.0
        self._comp_em = 0.0
        self._open_em = 0.0
        self._unsure_em = 0.0
        self._extract_em = 0.0
        self._arithmetic_em = 0.0
        self._arithmetic_open_em = 0.0
        self._judge_f1 = 0.0
        self._comp_f1 = 0.0
        self._open_f1 = 0.0
        self._unsure_f1 = 0.0
        self._extract_f1 = 0.0
        self._arithmetic_f1 = 0.0
        self._arithmetic_open_f1 = 0.0

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1}," \
               f"supporting_exact={self._supporting_exact},supporting_total={self._supporting_total}," \
               f"judge={self._judge_em, self._judge_f1}," \
               f"comp={self._comp_em, self._comp_f1}, " \
               f"open={self._open_em, self._open_f1}, " \
               f"unsure={self._unsure_em, self._unsure_f1}," \
               f"extract={self._extract_em, self._extract_f1}," \
               f"arithmetic={self._arithmetic_em, self._arithmetic_f1}," \
               f"arithmetic_open={self._arithmetic_open_em, self._arithmetic_open_f1})"