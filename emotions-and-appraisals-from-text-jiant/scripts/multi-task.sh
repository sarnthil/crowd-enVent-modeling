#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

source $HOME/venvs/jiant/bin/activate
timestamp=$(date -Iseconds)
cd $HOME/jiant
mkdir logs
time python crowdenvent_multi.py emo_cls suddenness_cls familiarity_cls predict_event_cls pleasantness_cls unpleasantness_cls goal_relevance_cls chance_responsblt_cls self_responsblt_cls other_responsblt_cls predict_conseq_cls goal_support_cls urgency_cls self_control_cls other_control_cls chance_control_cls accept_conseq_cls standards_cls social_norms_cls attention_cls not_consider_cls effort_cls >$HOME/jiant/logs/stdout-multi-cls-$timestamp.log 2>$HOME/jiant/logs/stderr-multi-cls-$timestamp.log
# time python crowdenvent_multi.py emo_cls suddenness_reg familiarity_reg predict_event_reg pleasantness_reg unpleasantness_reg goal_relevance_reg chance_responsblt_reg self_responsblt_reg other_responsblt_reg predict_conseq_reg goal_support_reg urgency_reg self_control_reg other_control_reg chance_control_reg accept_conseq_reg standards_reg social_norms_reg attention_reg not_consider_reg effort_reg >$HOME/jiant/logs/stdout-multi-reg-$timestamp.log 2>$HOME/jiant/logs/stderr-multi-reg-$timestamp.log
