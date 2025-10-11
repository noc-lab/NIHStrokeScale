import copy
from copy import deepcopy
from Neuro_SMEval.models import get_response_method
from Neuro_SMEval.chat_utils import (select_score_RankInterval_representatives,
                                     format_chat_LlamaTemplate2Text)
from Neuro_SMEval.config import default_CONFIG as CONFIG
import os
from transformers import AutoTokenizer
import random
import vllm
from vllm import LLM, SamplingParams
import numpy as np
import torch
import sys
import pandas as pd

# For token counting
import tiktoken
from llama_index.llms.openai import OpenAI

#Example usage

# Suppress logs from vllm, transformers, etc.
from vllm.logger import init_logger
init_logger("ERROR")  # Or "CRITICAL" for even stricter silence
# Import necessary libraries

import logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("vllm").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
# Clear the cache directory if it exists
_ = os.system('rm -rf /usr3/graduate/cuy/.cache/vllm/torch_compile_cache/ 2>/dev/null')


def get_vllm_outputs(outputs, IF_print = False, IF_return_gen = True): #possibly not useful
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if IF_print:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    if IF_print:
        print("-" * 80)

    if IF_return_gen:
        return [output.outputs[0].text for output in outputs][0]


class LLM_Instance_Pipe:
    def __init__(self,
                 model_id,
                 seed_gen,
                 api_key=None,
                 work_type=None,
                 api_base='http://localhost:8000/v1',
                 hf_home_cache_path=None
                 ):

        self.model_id = model_id
        self.model_name = self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id
        self.api_key = api_key
        self.api_base = api_base
        self.work_type = work_type
        self.hf_home_cache_path = hf_home_cache_path
        self.api = api_key is not None
        self.seed_gen = seed_gen
        self.local = not self.api

        if self.local:
            print("self.local is set to True, indicating local vLLM inference.")
            # Set HF cache
            if self.hf_home_cache_path is None:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                self.hf_home_cache_path = os.path.join(base_dir, ".cache")

            os.environ["HUGGINGFACE_HUB_CACHE"] = self.hf_home_cache_path
            os.environ["HF_HOME"] = self.hf_home_cache_path

            # Initialize tokenizer for chat formatting
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.hf_home_cache_path,
                use_fast=True
            )

            print("self.tokenizer has been set.")

            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            ######################################
            # Function to determine valid tensor parallel size
            def get_valid_tensor_parallel_size(max_gpus, num_heads):
                for tp_size in reversed(range(1, max_gpus + 1)):
                    if num_heads % tp_size == 0:
                        return tp_size
                return 1  # Fallback

            num_heads = 32  # mistral small has 32 heads
            num_gpus = torch.cuda.device_count()
            tp_size = get_valid_tensor_parallel_size(num_gpus, num_heads)
            #print(f"Using tensor_parallel_size={tp_size}")
            ######################################

            self.llm = LLM(model=self.model_id,
                      trust_remote_code=True,
                      enable_chunked_prefill=True,
                      max_model_len=CONFIG["vllm"]["max_model_len"],
                      tensor_parallel_size=tp_size,  # or more if using multi-GPU
                      gpu_memory_utilization=CONFIG["vllm"]["gpu_memory_utilization"],  # adjust as needed
                      seed=self.seed_gen, # Seed for reproducibility; if without setting then the default is 0
                      )

        else:
            # API-based mode (vLLM server or OpenAI)
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def get_respond(self,
                    messages,
                    seed, # this seed actually does not affect API-based calls--the seed for instance is set in the LLM_Instance_Pipe constructor
                    resp_temperature = None,
                    resp_max_tokens = None,
                    resp_top_p = None,
                    resp_API_or_locally_cache_model = None):

        # Seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        if self.api:
            # Query via OpenAI-compatible API (e.g., vLLM server or OpenAI)
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )
            return completion.choices[0].message.content.strip()

        else:
            # Local vLLM inference using SamplingParams from CONFIG
            cfg = CONFIG["local_agent"][self.work_type]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            if resp_temperature == None:
                resp_temperature = cfg.get("temperature", 0.7) if cfg["do_sample"] else 0.0
            if resp_top_p == None:
                resp_top_p = cfg.get("top_p", 0.95) if cfg["do_sample"] else 1.0
            if resp_max_tokens == None:
                resp_max_tokens = cfg.get("max_new_tokens", 512)

            vllm_sampling_params = SamplingParams(
                temperature=resp_temperature,
                top_p=resp_top_p,
                max_tokens=resp_max_tokens,
                stop=None  # You can add custom stop tokens here if needed
            )

            #deprecated: vLLM does not require a separate tokenizer initialization
            #outputs = self.llm.generate([prompt], sampling_params)
            #generated_text = outputs[0].outputs[0].text

            #outputs = self.llm.chat(messages,
            #                   sampling_params=vllm_sampling_params,
            #                   use_tqdm=False)
            # Generate outputs
            if 'llama' in self.model_name.lower():
                outputs = self.llm.generate([prompt],
                                            vllm_sampling_params)
            elif 'mistral' in self.model_name.lower():
                outputs = self.llm.chat(messages,
                                        sampling_params=vllm_sampling_params,
                                        use_tqdm=CONFIG["vllm"]["use_tqdm"])

            else:
                raise ValueError(f"Unsupported model type: {self.model_name}. Please use a llama or mistral model.")

            return get_vllm_outputs(outputs, IF_print=False, IF_return_gen=True)


def get_respond_pipe(answerer_model_pipe, task, history, temperature, max_tokens, work_type="generic"):
    """
    Get a direct answer from the LLM based on the provided history.
    """

    # Generate a response using the LLM instance
    bot_response = answerer_model_pipe.get_respond(history,
                                                   task.rd_seed,
                                                   temperature,
                                                   max_tokens
                                                   )

    return bot_response


def AggrPrompting_adju_AllAblExpr(task,
                                  i,
                                  list_InputNotes_settings = [
                                      "GS_Extraction_ExpandedAbbrev", # expr setting 1 #GS: Gold Standard #
                                      "GS_Extraction_NoExpandedAbbrev", #expr setting 2
                                      "AE_Extraction_ExpandedAbbrev", # expr setting 3 #AE: Auto Extraction, equivalent to Silver Standard
                                      "Note_ExpandedAbbrev", # expr setting 4
                                      "Note", # expr setting 5, full Notes without any abbreviation expansion
                                      "AE_Extraction_ExpandedAbbrev_250806fix", # expr setting 6
                                      "Note_ExpandedAbbrev_250806fix", # expr setting 7, 250806fix: full Notes with 250716 abbreviation expansion,
                                      "GS_Extraction_ExpandedAbbrev_250806fix", # expr setting 8, 250806fix: Gold Standard Extraction with 250716 abbreviation expansion,
                                ],
                                  list_OneTimePrompting_method = ["ZS_DA_AggrPrompt",
                                                                "FS_DA_AggrPrompt",
                                                                "ZS_CoT_AggrPrompt",
                                                                # "FS_CoT_AggrPrompt" #We don't have FS_CoT_AggrPrompt because we don't have human-annotated reasonings
                                                                ],
                                  list_num_FewShotExamples = [1, 2, 3], # by default, we can use 1-shot, 2-shot, and 3-shot examples for Few-Shot prompting
                                  answerer_model_pipe= None,
                                  ):

    # Initialize the output dictionary
    dict_output = {}
    inner_keys_DictOutput = [
        "ZS_DA_AggrPrompt",
        "FS_DA_AggrPrompt",
        "ZS_CoT_AggrPrompt",
        "FS_CoT_AggrPrompt"
    ]

    #outer_keys_DictOutput = deepcopy(list_InputNotes_settings)  # list_InputNotes_settings is exactly meant to be outer_keys_DictOutput
    inner_dict = {key: {} for key in inner_keys_DictOutput}
    print("inner_dict:", inner_dict)
    print("list_InputNotes_settings:", list_InputNotes_settings)
    for outer_key in list_InputNotes_settings:
        print("outer_key:", outer_key)
        dict_output[outer_key] = deepcopy(inner_dict)

    if task.Usage_GoldStandardAsContextualInput == 1:
        if task.ColName_ContextualInput != task.ColName_GoldStandard_Input:
            raise "task.ColName_ContextualInput not equal to task.ColName_GoldStandard_Input! Please confirm and fix the usage of GoldStandardAsContextualInput!"

    gold_input = task.data_TEST[i]["gold_input"]

    list_contextual_input_TEST = [task.data_TEST[i][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings] # We have various contextual inputs from abaltion settings

    Sample_ID = task.data_TEST[i]["Sample_ID"]
    gold_NIHSS_breakdown = task.data_TEST[i]["gold_NIHSS_breakdown"]
    gold_NIHSS = task.data_TEST[i]["gold_NIHSS"]
    #contextual_input = task.prompt_pool.contextual_input_declaration.format(contextual_input=gold_input)
    #print(contextual_input)

    if answerer_model_pipe is None:
        answerer_model_pipe = LLM_Instance_Pipe(model_id=task.base_model,
                                                seed_gen=task.rd_seed,
                                                work_type="generic",
                                                hf_home_cache_path=".cache"
                                                )
    else:
        print("Using provided answerer_model_pipe instance.")


    for InputNotes_settings in list_InputNotes_settings:

        for OneTimePrompting_method in list_OneTimePrompting_method:

            contextual_input = list_contextual_input_TEST[list_InputNotes_settings.index(InputNotes_settings)]

            if contextual_input == "" or contextual_input is None:
                raise "Invalid Input: contextual_input is empty or None for Sample_ID: {}".format(Sample_ID)
                break

                                
            elif OneTimePrompting_method == "ZS_DA_AggrPrompt":
                history_bot1 = [{'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7},
                                #{'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v8},
                                #{'role': 'user', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_prompt1_v1.format(contextual_input=contextual_input)},
                                {'role': 'user', 'content': contextual_input}
                                ]

            elif OneTimePrompting_method == "ZS_CoT_AggrPrompt":
                history_bot1 = [{'role': 'system', 'content': task.prompt_pool.ZS_CoT_AggrPrompt_RubricInstruction_sys_msg_v7},
                                #{'role': 'system', 'content': task.prompt_pool.ZS_CoT_AggrPrompt_RubricInstruction_sys_msg_v8},
                                #{'role': 'user', 'content': task.prompt_pool.ZS_CoT_AggrPrompt_RubricInstruction_prompt1_v1.format(contextual_input=contextual_input)},
                                {'role': 'user', 'content': contextual_input}
                                ]

            elif OneTimePrompting_method == "FS_DA_AggrPrompt":

                df_data = pd.DataFrame(task.data_FS)
                #print("df_data.shape:", df_data.shape)

                ##############################################################
                if 1 in list_num_FewShotExamples:
                    N_SelectN = 1
                    df_data_SelectN = select_score_RankInterval_representatives(df_data, N=N_SelectN, score_col='gold_NIHSS', seed=task.rd_seed)
                    #print(f"N_SelectN = {N_SelectN}, df_data_SelectN.shape:", df_data_SelectN.shape)
                    # For FS_DA_AggrPrompt, we need to use the contextual inputs from the FS dataset

                    #print("at N_SelectN = 1, df_data_SelectN.iloc[0]:", df_data_SelectN.iloc[0])
                    #print("list_InputNotes_settings:", list_InputNotes_settings)

                    idx_CurrentData_FS = 0
                    list_contextual_input_FS_DemoPair1 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair1 = list_contextual_input_FS_DemoPair1[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    history_bot_1shot = [#{'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7},
                                        #{'role': 'system', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v8},
                                         {'role': 'system', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_1shot.format(#contextual_input=contextual_input,
                                                                                                                                            contextual_input_example1=contextual_input_FS_DemoPair1,
                                                                                                                                            GS_adjudication_answer1=GS_score_FS_DemoPair1,
                                                                                                                                            )},
                                        {'role': 'user', 'content': contextual_input}
                    ]

                    list_FS_DA_SampleIDs_FSDemoPair_1shot = [Sample_ID_FS_DemoPair1]

                    list_FS_DA_SampleIDs_FSDemoPair_1shot = [int(elem) for elem in list_FS_DA_SampleIDs_FSDemoPair_1shot]  # Convert to int if needed
                ##############################################################
                ##############################################################
                if 2 in list_num_FewShotExamples:
                    N_SelectN = 2
                    df_data_SelectN = select_score_RankInterval_representatives(df_data, N=2, score_col='gold_NIHSS', seed=task.rd_seed)
                    #print(f"N_SelectN = {N_SelectN}, df_data_SelectN.shape:", df_data_SelectN.shape)
    
                    # For FS_DA_AggrPrompt, we need to use the contextual inputs from the FS dataset
                    idx_CurrentData_FS = 0
                    list_contextual_input_FS_DemoPair1 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair1 = list_contextual_input_FS_DemoPair1[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']
    
                    idx_CurrentData_FS = 1
                    list_contextual_input_FS_DemoPair2 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair2 = list_contextual_input_FS_DemoPair2[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']
    
                    history_bot_2shot = [#{'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7},
                                        #{'role': 'system', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v8},
                                        {'role': 'user', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_2shot.format(#contextual_input=contextual_input,
                                                                                                                                            contextual_input_example1=contextual_input_FS_DemoPair1,
                                                                                                                                            GS_adjudication_answer1=GS_score_FS_DemoPair1,
                                                                                                                                            contextual_input_example2=contextual_input_FS_DemoPair2,
                                                                                                                                            GS_adjudication_answer2=GS_score_FS_DemoPair2,
                                                                                                                                            )},
                                        {'role': 'user', 'content': contextual_input}
                    ]

                    list_FS_DA_SampleIDs_FSDemoPair_2shot = [Sample_ID_FS_DemoPair1, Sample_ID_FS_DemoPair2]

                    list_FS_DA_SampleIDs_FSDemoPair_2shot = [int(elem) for elem in list_FS_DA_SampleIDs_FSDemoPair_2shot]  # Convert to int if needed
                ##############################################################
                ##############################################################
                if 3 in list_num_FewShotExamples:
                    N_SelectN = 3
                    df_data_SelectN = select_score_RankInterval_representatives(df_data, N=3, score_col='gold_NIHSS', seed=task.rd_seed)
                    #print(f"N_SelectN = {N_SelectN}, df_data_SelectN.shape:", df_data_SelectN.shape)
                    # For FS_DA_AggrPrompt, we need to use the contextual inputs from the FS dataset
                    idx_CurrentData_FS = 0
                    list_contextual_input_FS_DemoPair1 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair1 = list_contextual_input_FS_DemoPair1[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 1
                    list_contextual_input_FS_DemoPair2 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair2 = list_contextual_input_FS_DemoPair2[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 2
                    list_contextual_input_FS_DemoPair3 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair3 = list_contextual_input_FS_DemoPair3[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair3 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair3 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    history_bot_3shot = [#'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7},
                                        #{'role': 'system', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v8},
                                        {'role': 'user', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_3shot.format(#contextual_input=contextual_input,
                                                                                                                                            contextual_input_example1=contextual_input_FS_DemoPair1,
                                                                                                                                            GS_adjudication_answer1=GS_score_FS_DemoPair1,
                                                                                                                                            contextual_input_example2=contextual_input_FS_DemoPair2,
                                                                                                                                            GS_adjudication_answer2=GS_score_FS_DemoPair2,
                                                                                                                                            contextual_input_example3=contextual_input_FS_DemoPair3,
                                                                                                                                            GS_adjudication_answer3=GS_score_FS_DemoPair3,
                                                                                                                                            )},
                                        {'role': 'user', 'content': contextual_input}
                    ]

                    list_FS_DA_SampleIDs_FSDemoPair_3shot = [Sample_ID_FS_DemoPair1,
                                                            Sample_ID_FS_DemoPair2,
                                                            Sample_ID_FS_DemoPair3]

                    list_FS_DA_SampleIDs_FSDemoPair_3shot = [int(elem) for elem in list_FS_DA_SampleIDs_FSDemoPair_3shot]  # Convert to int if needed
                ##############################################################
                ##############################################################
                if 4 in list_num_FewShotExamples:
                    N_SelectN = 4
                    df_data_SelectN = select_score_RankInterval_representatives(df_data, N=4, score_col='gold_NIHSS', seed=task.rd_seed)
                    #print(f"N_SelectN = {N_SelectN}, df_data_SelectN.shape:", df_data_SelectN.shape)
                    # For FS_DA_AggrPrompt, we need to use the contextual inputs from the FS dataset
                    idx_CurrentData_FS = 0
                    list_contextual_input_FS_DemoPair1 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair1 = list_contextual_input_FS_DemoPair1[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 1
                    list_contextual_input_FS_DemoPair2 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair2 = list_contextual_input_FS_DemoPair2[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 2
                    list_contextual_input_FS_DemoPair3 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair3 = list_contextual_input_FS_DemoPair3[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair3 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair3 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 3
                    list_contextual_input_FS_DemoPair4 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair4 = list_contextual_input_FS_DemoPair4[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair4 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair4 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    history_bot_4shot = [#{'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7},
                                        #{'role': 'system', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v8},
                                        {'role': 'user', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_4shot.format(#contextual_input=contextual_input,
                                                                                                                                            contextual_input_example1=contextual_input_FS_DemoPair1,
                                                                                                                                            GS_adjudication_answer1=GS_score_FS_DemoPair1,
                                                                                                                                            contextual_input_example2=contextual_input_FS_DemoPair2,
                                                                                                                                            GS_adjudication_answer2=GS_score_FS_DemoPair2,
                                                                                                                                            contextual_input_example3=contextual_input_FS_DemoPair3,
                                                                                                                                            GS_adjudication_answer3=GS_score_FS_DemoPair3,
                                                                                                                                            contextual_input_example4=contextual_input_FS_DemoPair4,
                                                                                                                                            GS_adjudication_answer4=GS_score_FS_DemoPair4,
                                                                                                                                            )},
                                        {'role': 'user', 'content': contextual_input}
                                         ]

                    list_FS_DA_SampleIDs_FSDemoPair_4shot = [Sample_ID_FS_DemoPair1,
                                                            Sample_ID_FS_DemoPair2,
                                                            Sample_ID_FS_DemoPair3,
                                                            Sample_ID_FS_DemoPair4]

                    list_FS_DA_SampleIDs_FSDemoPair_4shot = [int(elem) for elem in list_FS_DA_SampleIDs_FSDemoPair_4shot]  # Convert to int if needed
                ##############################################################
                ##############################################################
                if 5 in list_num_FewShotExamples:
                    N_SelectN = 5
                    df_data_SelectN = select_score_RankInterval_representatives(df_data, N=5, score_col='gold_NIHSS', seed=task.rd_seed)
                    #print(f"N_SelectN = {N_SelectN}, df_data_SelectN.shape:", df_data_SelectN.shape)
                    # For FS_DA_AggrPrompt, we need to use the contextual inputs from the FS dataset

                    idx_CurrentData_FS = 0
                    list_contextual_input_FS_DemoPair1 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair1 = list_contextual_input_FS_DemoPair1[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair1 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 1
                    list_contextual_input_FS_DemoPair2 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair2 = list_contextual_input_FS_DemoPair2[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair2 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 2
                    list_contextual_input_FS_DemoPair3 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair3 = list_contextual_input_FS_DemoPair3[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair3 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair3 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 3
                    list_contextual_input_FS_DemoPair4 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair4 = list_contextual_input_FS_DemoPair4[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair4 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair4 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    idx_CurrentData_FS = 4
                    list_contextual_input_FS_DemoPair5 = [df_data_SelectN.iloc[idx_CurrentData_FS][InputNotes_settings] for InputNotes_settings in list_InputNotes_settings]
                    contextual_input_FS_DemoPair5 = list_contextual_input_FS_DemoPair5[list_InputNotes_settings.index(InputNotes_settings)]
                    GS_score_FS_DemoPair5 = df_data_SelectN.iloc[idx_CurrentData_FS]['gold_NIHSS_breakdown']
                    Sample_ID_FS_DemoPair5 = df_data_SelectN.iloc[idx_CurrentData_FS]['Sample_ID']

                    history_bot_5shot = [#{'role': 'system', 'content': task.prompt_pool.ZS_DA_AggrPrompt_RubricInstruction_sys_msg_v7},
                                        #{'role': 'system', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v8},
                                        {'role': 'user', 'content': task.prompt_pool.FS_DA_AggrPrompt_RubricInstruction_sys_msg_v7_5shot.format(#contextual_input=contextual_input,
                                                                                                                                            contextual_input_example1=contextual_input_FS_DemoPair1,
                                                                                                                                            GS_adjudication_answer1=GS_score_FS_DemoPair1,
                                                                                                                                            contextual_input_example2=contextual_input_FS_DemoPair2,
                                                                                                                                            GS_adjudication_answer2=GS_score_FS_DemoPair2,
                                                                                                                                            contextual_input_example3=contextual_input_FS_DemoPair3,
                                                                                                                                            GS_adjudication_answer3=GS_score_FS_DemoPair3,
                                                                                                                                            contextual_input_example4=contextual_input_FS_DemoPair4,
                                                                                                                                            GS_adjudication_answer4=GS_score_FS_DemoPair4,
                                                                                                                                            contextual_input_example5=contextual_input_FS_DemoPair5,
                                                                                                                                            GS_adjudication_answer5=GS_score_FS_DemoPair5,
                                                                                                                                            )},
                                        {'role': 'user', 'content': contextual_input}
                                         ]

                    list_FS_DA_SampleIDs_FSDemoPair_5shot = [Sample_ID_FS_DemoPair1,
                                                            Sample_ID_FS_DemoPair2,
                                                            Sample_ID_FS_DemoPair3,
                                                            Sample_ID_FS_DemoPair4,
                                                            Sample_ID_FS_DemoPair5]

                    list_FS_DA_SampleIDs_FSDemoPair_5shot = [int(elem) for elem in list_FS_DA_SampleIDs_FSDemoPair_5shot]  # Convert to int if needed
                ##############################################################
                ##############################################################


            elif OneTimePrompting_method == "FS_CoT_AggrPrompt":
                #history_bot1 = [{'role': 'system', 'content': task.prompt_pool.FS_CoT_AggrPrompt_RubricInstruction_sys_msg_v8},
                #                {'role': 'user', 'content': task.prompt_pool.FS_CoT_AggrPrompt_RubricInstruction_prompt1_v1.format(contextual_input=contextual_input)}]
                raise "Not Implemented Error: FS_CoT_AggrPrompt is an invalid One-Time Prompting method"
            else:
                raise "Not Implemented Error: Invalid One-Time Prompting method"

            ########################################################################################
            ########Feed the history_bot1 to the answerer_model_pipe to get the bot response########
            ########################################################################################

            # Set up the one common encoding for token counting
            encoding = tiktoken.encoding_for_model(task.embed_model_DefaultCommon)  # The default embedding model is gpt-4o

            if "fs" not in OneTimePrompting_method.lower():
                bot1_response = get_respond_pipe(answerer_model_pipe,
                                                 task,
                                                 history_bot1,
                                                 task.temperature,
                                                 task.max_tokens,
                                                 "generic"  # work_type can be "generic" by default
                                                 )

                #count the length of input tokens
                history_bot1_2Text = format_chat_LlamaTemplate2Text(history_bot1)
                history_bot1_TokenCount = len(encoding.encode(history_bot1_2Text))
                bot1_response_TokenCount = len(encoding.encode(bot1_response))

            else: # "fs" in OneTimePrompting_method.lower(), aka Few-Shot
                if 1 in list_num_FewShotExamples:
                    bot_response_1shot = get_respond_pipe(answerer_model_pipe,
                                                     task,
                                                     history_bot_1shot,
                                                     task.temperature,
                                                     task.max_tokens,
                                                     "generic"  # work_type can be "generic" by default
                                                     )
                    #count the length of input tokens
                    #encoding = tiktoken.encoding_for_model(task.base_model)
                    history_bot_1shot_2Text = format_chat_LlamaTemplate2Text(history_bot_1shot)
                    #encoding = tiktoken.encoding_for_model(task.embed_model_DefaultCommon) #The default embedding model is gpt-4o
                    history_bot_1shot_TokenCount = len(encoding.encode(history_bot_1shot_2Text))
                    bot_response_1shot_TokenCount = len(encoding.encode(bot_response_1shot))

                if 2 in list_num_FewShotExamples:
                    bot_response_2shot = get_respond_pipe(answerer_model_pipe,
                                                     task,
                                                     history_bot_2shot,
                                                     task.temperature,
                                                     task.max_tokens,
                                                     "generic"  # work_type can be "generic" by default
                                                     )
                    #count the length of input tokens
                    #encoding = tiktoken.encoding_for_model(task.base_model)
                    history_bot_2shot_2Text = format_chat_LlamaTemplate2Text(history_bot_2shot)
                    #encoding = tiktoken.encoding_for_model(task.embed_model_DefaultCommon) #The default embedding model is gpt-4o
                    history_bot_2shot_TokenCount = len(encoding.encode(history_bot_2shot_2Text))
                    bot_response_2shot_TokenCount = len(encoding.encode(bot_response_2shot))

                if 3 in list_num_FewShotExamples:
                    bot_response_3shot = get_respond_pipe(answerer_model_pipe,
                                                        task,
                                                        history_bot_3shot,
                                                        task.temperature,
                                                        task.max_tokens,
                                                        "generic"  # work_type can be "generic" by default
                                                        )
                    #count the length of input tokens
                    #encoding = tiktoken.encoding_for_model(task.base_model)
                    history_bot_3shot_2Text = format_chat_LlamaTemplate2Text(history_bot_3shot)
                    #encoding = tiktoken.encoding_for_model(task.embed_model_DefaultCommon) #The default embedding model is gpt-4o
                    history_bot_3shot_TokenCount = len(encoding.encode(history_bot_3shot_2Text))
                    bot_response_3shot_TokenCount = len(encoding.encode(bot_response_3shot))

                if 4 in list_num_FewShotExamples:
                    bot_response_4shot = get_respond_pipe(answerer_model_pipe,
                                                        task,
                                                        history_bot_4shot,
                                                        task.temperature,
                                                        task.max_tokens,
                                                        "generic"  # work_type can be "generic" by default
                                                        )
                    #count the length of input tokens
                    #encoding = tiktoken.encoding_for_model(task.base_model)
                    history_bot_4shot_2Text = format_chat_LlamaTemplate2Text(history_bot_4shot)
                    #encoding = tiktoken.encoding_for_model(task.embed_model_DefaultCommon) #The default embedding model is gpt-4o
                    history_bot_4shot_TokenCount = len(encoding.encode(history_bot_4shot_2Text))
                    bot_response_4shot_TokenCount = len(encoding.encode(bot_response_4shot))

                if 5 in list_num_FewShotExamples:
                    bot_response_5shot = get_respond_pipe(answerer_model_pipe,
                                                        task,
                                                        history_bot_5shot,
                                                        task.temperature,
                                                        task.max_tokens,
                                                        "generic"  # work_type can be "generic" by default
                                                        )
                    #count the length of input tokens
                    #encoding = tiktoken.encoding_for_model(task.base_model)
                    history_bot_5shot_2Text = format_chat_LlamaTemplate2Text(history_bot_5shot)
                    #encoding = tiktoken.encoding_for_model(task.embed_model_DefaultCommon) #The default embedding model is gpt-4o
                    history_bot_5shot_TokenCount = len(encoding.encode(history_bot_5shot_2Text))
                    bot_response_5shot_TokenCount = len(encoding.encode(bot_response_5shot))

            ########################################################################################
            ########Feed the history_bot1 to the answerer_model_pipe to get the bot response########
            ########################################################################################

            if task.Usage_GoldStandardAsContextualInput == 1:
                if 'fs' not in OneTimePrompting_method.lower():
                    dict_response_save = {'Sample_ID': Sample_ID,
                                          'gold_input': gold_input,
                                          'bot_response': bot1_response,
                                          'gold_NIHSS': gold_NIHSS,
                                          'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                          'input_TokenCount': history_bot1_TokenCount,
                                          'bot_response_TokenCount': bot1_response_TokenCount}
                else: # "fs" in OneTimePrompting_method.lower(), aka Few-Shot
                    dict_response_save = {}
                    if 1 in list_num_FewShotExamples:
                        dict_1shot = {'Sample_ID': Sample_ID, 'gold_input': gold_input, 'bot_response': bot_response_1shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_1shot),
                                      'input_TokenCount': history_bot_1shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_1shot_TokenCount}
                        dict_response_save["dict_FS_1shot"] = dict_1shot
                    if 2 in list_num_FewShotExamples:
                        dict_2shot = {'Sample_ID': Sample_ID, 'gold_input': gold_input, 'bot_response': bot_response_2shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_2shot),
                                      'input_TokenCount': history_bot_2shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_2shot_TokenCount}
                        dict_response_save["dict_FS_2shot"] = dict_2shot
                    if 3 in list_num_FewShotExamples:
                        dict_3shot = {'Sample_ID': Sample_ID, 'gold_input': gold_input, 'bot_response': bot_response_3shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_3shot),
                                      'input_TokenCount': history_bot_3shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_3shot_TokenCount}
                        dict_response_save["dict_FS_3shot"] = dict_3shot
                    if 4 in list_num_FewShotExamples:
                        dict_4shot = {'Sample_ID': Sample_ID, 'gold_input': gold_input, 'bot_response': bot_response_4shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_4shot),
                                      'input_TokenCount': history_bot_4shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_4shot_TokenCount}
                        dict_response_save["dict_FS_4shot"] = dict_4shot
                    if 5 in list_num_FewShotExamples:
                        dict_5shot = {'Sample_ID': Sample_ID, 'gold_input': gold_input, 'bot_response': bot_response_5shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_5shot),
                                      'input_TokenCount': history_bot_5shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_5shot_TokenCount}
                        dict_response_save["dict_FS_5shot"] = dict_5shot

                    #dict_response_save = {"dict_FS_1shot": dict_1shot, "dict_FS_2shot": dict_2shot, "dict_FS_3shot": dict_3shot, "dict_FS_4shot": dict_4shot,"dict_FS_5shot": dict_5shot,}

            else: # aka task.Usage_GoldStandardAsContextualInput != 1
                if 'fs' not in OneTimePrompting_method.lower():
                    dict_response_save = {'Sample_ID': Sample_ID,
                                          'contextual_input': contextual_input,
                                          'gold_input': gold_input,
                                          'bot_response': bot1_response,
                                          'gold_NIHSS': gold_NIHSS,
                                          'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                          'input_TokenCount': history_bot1_TokenCount,
                                          'bot_response_TokenCount': bot1_response_TokenCount}
                else: # "fs" in OneTimePrompting_method.lower(), aka Few-Shot

                    dict_response_save = {}

                    if 1 in list_num_FewShotExamples:
                        dict_1shot = {'Sample_ID': Sample_ID,
                                      'contextual_input': contextual_input,
                                      'gold_input': gold_input,
                                      'bot_response': bot_response_1shot,
                                      'gold_NIHSS': gold_NIHSS,
                                      'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_1shot),
                                      'input_TokenCount': history_bot_1shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_1shot_TokenCount}
                        dict_response_save["dict_FS_1shot"] = dict_1shot
                    if 2 in list_num_FewShotExamples:
                        dict_2shot = {'Sample_ID': Sample_ID, 'contextual_input': contextual_input, 'gold_input': gold_input, 'bot_response': bot_response_2shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_2shot),
                                      'input_TokenCount': history_bot_2shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_2shot_TokenCount}
                        dict_response_save["dict_FS_2shot"] = dict_2shot
                    if 3 in list_num_FewShotExamples:
                        dict_3shot = {'Sample_ID': Sample_ID, 'contextual_input': contextual_input, 'gold_input': gold_input, 'bot_response': bot_response_3shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_3shot),
                                      'input_TokenCount': history_bot_3shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_3shot_TokenCount}
                        dict_response_save["dict_FS_3shot"] = dict_3shot
                    if 4 in list_num_FewShotExamples:
                        dict_4shot = {'Sample_ID': Sample_ID, 'contextual_input': contextual_input, 'gold_input': gold_input, 'bot_response': bot_response_4shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_4shot),
                                      'input_TokenCount': history_bot_4shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_4shot_TokenCount}
                        dict_response_save["dict_FS_4shot"] = dict_4shot
                    if 5 in list_num_FewShotExamples:
                        dict_5shot = {'Sample_ID': Sample_ID, 'contextual_input': contextual_input, 'gold_input': gold_input, 'bot_response': bot_response_5shot, 'gold_NIHSS': gold_NIHSS, 'gold_NIHSS_breakdown': gold_NIHSS_breakdown,
                                      'list_FS_DA_SampleIDs_FSDemoPair': str(list_FS_DA_SampleIDs_FSDemoPair_5shot),
                                      'input_TokenCount': history_bot_5shot_TokenCount,
                                      'bot_response_TokenCount': bot_response_5shot_TokenCount}
                        dict_response_save["dict_FS_5shot"] = dict_5shot

                    #dict_response_save = {"dict_FS_1shot": dict_1shot, "dict_FS_2shot": dict_2shot, "dict_FS_3shot": dict_3shot, "dict_FS_4shot": dict_4shot,"dict_FS_5shot": dict_5shot,}

            #save dict_response_save into the corresponding slot in dict_output
            dict_output[InputNotes_settings][OneTimePrompting_method] = copy.deepcopy(dict_response_save)

    #return the dict_output
    return dict_output
