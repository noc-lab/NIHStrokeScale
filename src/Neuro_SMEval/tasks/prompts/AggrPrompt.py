from Neuro_SMEval.tasks.prompts.general import *

#############################
#Useful prompts start here...
ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7 = '''
Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores.
1a (Level of Consciousness): 	#
1b (Answers Questions): 	#
1c (Follows Commands): 	#
2 (Best Gaze): 	#
3 (Visual Fields): 	#
4 (Facial Palsy): 	#
5a (Motor Arm, Left): 	# 
5b (Motor Arm, Right): 	# 
6a (Motor Leg, Left): 	#
6b (Motor Leg, Right): 	#
7 (Limb Ataxia): 	#
8 (Sensory): 	#
9 (Best Language): 	#
10 (Dysarthria): 	#
11 (Extinction and Inattention): 	# 
Total NIH Stroke Scale Score: 	#

The remaining part of the conversation is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''


ZS_CoT_AggrPrompt_RubricInstruction_sys_msg_v7 = '''

You MUST state your reasoning step-by-step for each itemized examination before providing your final answer WITH A REQUIRED FORMAT. 

Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. 

If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. 

Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores. 

You will use the NIH stroke scale template to respond as your final answer when given exam text replacing # with your score:

(Make sure in your answer you strictly preserve the 'Reasoning: ...' and 'Scoring: <[<#>]>' as they are, and only replace '...' and '#' in your responses.)

1a (Level of Consciousness):\nReasoning: ...\nScoring: <[<#>]>\n
1b (Answers Questions):\nReasoning: ...\nScoring: <[<#>]>\n
1c (Follows Commands):\nReasoning: ...\nScoring: <[<#>]>\n
2 (Best Gaze):\nReasoning: ...\nScoring: <[<#>]>\n
3 (Visual Fields):\nReasoning: ...\nScoring: <[<#>]>\n
4 (Facial Palsy):\nReasoning: ...\nScoring: <[<#>]>\n
5a (Motor Arm, Left):\nReasoning: ...\nScoring: <[<#>]>\n
5b (Motor Arm, Right):\nReasoning: ...\nScoring: <[<#>]>\n
6a (Motor Leg, Left):\nReasoning: ...\nScoring: <[<#>]>\n
6b (Motor Leg, Right):\nReasoning: ...\nScoring: <[<#>]>\n
7 (Limb Ataxia):\nReasoning: ...\nScoring: <[<#>]>\n
8 (Sensory):\nReasoning: ...\nScoring: <[<#>]>\n
9 (Best Language):\nReasoning: ...\nScoring: <[<#>]>\n
10 (Dysarthria):\nReasoning: ...\nScoring: <[<#>]>\n
11 (Extinction and Inattention):\nReasoning: ...\nScoring: <[<#>]>\n
Total NIH Stroke Scale Score: <[<#>]>

For example: 
1a (Level of Consciousness):
Reasoning: The patient is not alert, arousable by minor stimulation to obey, answer, respond. This indicates a level of consciousness of 1.
Scoring: <[<1>]>

The remaining part of the conversation is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''


FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_1shot = '''
Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores. 
1a (Level of Consciousness): 	#
1b (Answers Questions): 	#
1c (Follows Commands): 	#
2 (Best Gaze): 	#
3 (Visual Fields): 	#
4 (Facial Palsy): 	#
5a (Motor Arm, Left): 	# 
5b (Motor Arm, Right): 	# 
6a (Motor Leg, Left): 	#
6b (Motor Leg, Right): 	#
7 (Limb Ataxia): 	#
8 (Sensory): 	#
9 (Best Language): 	#
10 (Dysarthria): 	#
11 (Extinction and Inattention): 	# 
Total NIH Stroke Scale Score: 	#

Some practical example Context-Answer pairs will be provided to illustrate this clinical task.

Pair1 - Example Contextual Input:
{contextual_input_example1}

Pair1 - Gold Standard Answer:
{GS_adjudication_answer1}

The remaining part of this prompt is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''


FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_2shot = '''
Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores. 
1a (Level of Consciousness): 	#
1b (Answers Questions): 	#
1c (Follows Commands): 	#
2 (Best Gaze): 	#
3 (Visual Fields): 	#
4 (Facial Palsy): 	#
5a (Motor Arm, Left): 	# 
5b (Motor Arm, Right): 	# 
6a (Motor Leg, Left): 	#
6b (Motor Leg, Right): 	#
7 (Limb Ataxia): 	#
8 (Sensory): 	#
9 (Best Language): 	#
10 (Dysarthria): 	#
11 (Extinction and Inattention): 	# 
Total NIH Stroke Scale Score: 	#

Some practical example Context-Answer pairs will be provided to illustrate this clinical task.

Pair1 - Example Contextual Input:
{contextual_input_example1}

Pair1 - Gold Standard Answer:
{GS_adjudication_answer1}

Pair2 - Example Contextual Input:
{contextual_input_example2}

Pair2 - Gold Standard Answer:
{GS_adjudication_answer2}

The remaining part of this prompt is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''


FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_3shot = '''
Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores. 
1a (Level of Consciousness): 	#
1b (Answers Questions): 	#
1c (Follows Commands): 	#
2 (Best Gaze): 	#
3 (Visual Fields): 	#
4 (Facial Palsy): 	#
5a (Motor Arm, Left): 	# 
5b (Motor Arm, Right): 	# 
6a (Motor Leg, Left): 	#
6b (Motor Leg, Right): 	#
7 (Limb Ataxia): 	#
8 (Sensory): 	#
9 (Best Language): 	#
10 (Dysarthria): 	#
11 (Extinction and Inattention): 	# 
Total NIH Stroke Scale Score: 	#

Some practical example Context-Answer pairs will be provided to illustrate this clinical task.

Pair1 - Example Contextual Input:
{contextual_input_example1}

Pair1 - Gold Standard Answer:
{GS_adjudication_answer1}

Pair2 - Example Contextual Input:
{contextual_input_example2}

Pair2 - Gold Standard Answer:
{GS_adjudication_answer2}

Pair3 - Example Contextual Input:
{contextual_input_example3}

Pair3 - Gold Standard Answer:
{GS_adjudication_answer3}

The remaining part of this prompt is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''


FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_4shot = '''
Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores. 
1a (Level of Consciousness): 	#
1b (Answers Questions): 	#
1c (Follows Commands): 	#
2 (Best Gaze): 	#
3 (Visual Fields): 	#
4 (Facial Palsy): 	#
5a (Motor Arm, Left): 	# 
5b (Motor Arm, Right): 	# 
6a (Motor Leg, Left): 	#
6b (Motor Leg, Right): 	#
7 (Limb Ataxia): 	#
8 (Sensory): 	#
9 (Best Language): 	#
10 (Dysarthria): 	#
11 (Extinction and Inattention): 	# 
Total NIH Stroke Scale Score: 	#

Some practical example Context-Answer pairs will be provided to illustrate this clinical task.

Pair1 - Example Contextual Input:
{contextual_input_example1}

Pair1 - Gold Standard Answer:
{GS_adjudication_answer1}

Pair2 - Example Contextual Input:
{contextual_input_example2}

Pair2 - Gold Standard Answer:
{GS_adjudication_answer2}

Pair3 - Example Contextual Input:
{contextual_input_example3}

Pair3 - Gold Standard Answer:
{GS_adjudication_answer3}

Pair4 - Example Contextual Input:
{contextual_input_example4}

Pair4 - Gold Standard Answer:
{GS_adjudication_answer4}

The remaining part of this prompt is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''


FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_5shot = '''
Grade the following text numerically using the components of the NIH Stroke Scale and provide a sum. If an item is normal, state '0'. Absent items = 0. 1a = 0 unless stated otherwise. Remember to evaluate 1a, 1b, 1c and coma or unresponsive patients will score 1a = 3, 1b = 2, 1c = 2, 9 = 3. Remember to specify items 5a, 5b, 6a, 6b and that for motor exam: no movement (in text 0-1) = 4, barely moves (or in the text as 2) = 3, drifts to bed (in text as 3) = 2, mild weakness (in text as 4) = 1, and normal or full or without drift (in text 4+ to 5) = 0. Item 9: if expressive deficit alone = 1, severe expressive deficit or with inability to understand = 2, mute =3 (then item 10=2). Report your results strictly within the following template replacing # with your scores. 
1a (Level of Consciousness): 	#
1b (Answers Questions): 	#
1c (Follows Commands): 	#
2 (Best Gaze): 	#
3 (Visual Fields): 	#
4 (Facial Palsy): 	#
5a (Motor Arm, Left): 	# 
5b (Motor Arm, Right): 	# 
6a (Motor Leg, Left): 	#
6b (Motor Leg, Right): 	#
7 (Limb Ataxia): 	#
8 (Sensory): 	#
9 (Best Language): 	#
10 (Dysarthria): 	#
11 (Extinction and Inattention): 	# 
Total NIH Stroke Scale Score: 	#

Some practical example Context-Answer pairs will be provided to illustrate this clinical task.

Pair1 - Example Contextual Input:
{contextual_input_example1}

Pair1 - Gold Standard Answer:
{GS_adjudication_answer1}

Pair2 - Example Contextual Input:
{contextual_input_example2}

Pair2 - Gold Standard Answer:
{GS_adjudication_answer2}

Pair3 - Example Contextual Input:
{contextual_input_example3}

Pair3 - Gold Standard Answer:
{GS_adjudication_answer3}

Pair4 - Example Contextual Input:
{contextual_input_example4}

Pair4 - Gold Standard Answer:
{GS_adjudication_answer4}

Pair5 - Example Contextual Input:
{contextual_input_example5}

Pair5 - Gold Standard Answer:
{GS_adjudication_answer5}

The remaining part of this prompt is a practical clinical record, and you MUST refer to it to grade the NIHSS itemized & total scores. Please ensure you follow the above 12-line template and rubric strictly.
'''

