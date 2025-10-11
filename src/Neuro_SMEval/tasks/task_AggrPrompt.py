from Neuro_SMEval.chat_utils import import_prompts_by_task
import json
import pandas as pd


class AggrPrompt_Task_AllAblExpr:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        #self.prompt_pool = import_prompts_by_task("AggrPrompt")
        self.prompt_pool = import_prompts_by_task("AggrPrompt") #v7: fixed version, rather than using AggrPrompt.py
        self.data_TEST = self.load_dataset(args.dataset_TEST,
                                      args.ColName_GoldStandardInput, #should coincide with ColName_GS_Extraction_ExpandedAbbrev->only for sanity check
                                      args.ColName_GS_Extraction_ExpandedAbbrev, # expr setting 1 #GS: Gold Standard #
                                      args.ColName_GS_Extraction_NoExpandedAbbrev, #expr setting 2
                                      args.ColName_AE_Extraction_NoExpandedAbbrev, # expr setting 3 #AE: Auto Extraction, equivalent to Silver Standard
                                      args.ColName_Note_ExpandedAbbrev, # expr setting 4
                                      args.ColName_Note, # expr setting 5, full Notes without any abbreviation expansion
                                      args.ColName_AE_Extraction_ExpandedAbbrev_250806fix, # expr setting 6
                                      args.ColName_Note_ExpandedAbbrev_250806fix, # expr setting 7, 250806fix: full Notes with 250716 abbreviation expansion,
                                      args.ColName_GS_Extraction_ExpandedAbbrev_250806fix, # expr setting 8, 250806fix: Gold Standard Extraction with 250716 abbreviation expansion,
                                      args.ColName_GoldStandard_NIHSS, #Gold Standard NIHSS score from human experts
                                      args.ColName_GoldStandard_NIHSS_breakdown, #Gold Standard NIHSS breakdown from human experts
                                      )
        self.data_TRAIN = self.load_dataset(args.dataset_TRAIN,
                                      args.ColName_GoldStandardInput,
                                      args.ColName_GS_Extraction_ExpandedAbbrev,
                                      args.ColName_GS_Extraction_NoExpandedAbbrev,
                                      args.ColName_AE_Extraction_NoExpandedAbbrev,
                                      args.ColName_Note_ExpandedAbbrev,
                                      args.ColName_Note,
                                      args.ColName_AE_Extraction_ExpandedAbbrev_250806fix,
                                      args.ColName_Note_ExpandedAbbrev_250806fix,
                                      args.ColName_GS_Extraction_ExpandedAbbrev_250806fix,
                                      args.ColName_GoldStandard_NIHSS,
                                      args.ColName_GoldStandard_NIHSS_breakdown,
                                      )
        self.data_FS = self.load_dataset(args.dataset_FS,
                                      args.ColName_GoldStandardInput,
                                      args.ColName_GS_Extraction_ExpandedAbbrev,
                                      args.ColName_GS_Extraction_NoExpandedAbbrev,
                                      args.ColName_AE_Extraction_NoExpandedAbbrev,
                                      args.ColName_Note_ExpandedAbbrev,
                                      args.ColName_Note,
                                      args.ColName_AE_Extraction_ExpandedAbbrev_250806fix,
                                      args.ColName_Note_ExpandedAbbrev_250806fix,
                                      args.ColName_GS_Extraction_ExpandedAbbrev_250806fix,
                                      args.ColName_GoldStandard_NIHSS,
                                      args.ColName_GoldStandard_NIHSS_breakdown,
                                      )

    def load_dataset(self,
                    dataset_name_or_path,
                    ColName_GoldStandardInput,
                    ColName_GS_Extraction_ExpandedAbbrev,
                    ColName_GS_Extraction_NoExpandedAbbrev,
                    ColName_AE_Extraction_NoExpandedAbbrev,
                    ColName_Note_ExpandedAbbrev,
                    ColName_Note,
                    ColName_AE_Extraction_ExpandedAbbrev_250806fix,
                    ColName_Note_ExpandedAbbrev_250806fix,
                    ColName_GS_Extraction_ExpandedAbbrev_250806fix,
                    ColName_GoldStandard_NIHSS,
                    ColName_GoldStandard_NIHSS_breakdown
    ):

        if dataset_name_or_path in ['medqa']: #placeholder for potential future PUBLIC benchmark datasets
            test_qa = []
            #examplers = []

            test_path = f'../data/{dataset_name_or_path}/test.jsonl'
            with open(test_path, 'r') as file:
                for line in file:
                    test_qa.append(json.loads(line))

            return test_qa

        elif dataset_name_or_path.endswith('.csv'):
            print(f"Loading dataset from {dataset_name_or_path}...")
            df = pd.read_csv(dataset_name_or_path)
            if 'test' in dataset_name_or_path.lower():
                print(f"~!!!!Test dataset name or path: {dataset_name_or_path}")
                print("~!!!!Test dataset columns in the dataset:", df.columns)
                print(f"~!!!!Test dataset shape of the dataset: {df.shape}")

            try:
                output =  [
                    {
                    "Sample_ID": row['Sample_ID'],
                    "gold_input": row[ColName_GoldStandardInput],
                    "GS_Extraction_ExpandedAbbrev": row[ColName_GS_Extraction_ExpandedAbbrev], # expr setting 1
                    "GS_Extraction_NoExpandedAbbrev": row[ColName_GS_Extraction_NoExpandedAbbrev], # expr setting 2
                    "AE_Extraction_NoExpandedAbbrev": row[ColName_AE_Extraction_NoExpandedAbbrev], # expr setting 3
                    "Note_ExpandedAbbrev": row[ColName_Note_ExpandedAbbrev], # expr setting 4
                    "Note": row["Note"], # expr setting 5
                    "AE_Extraction_ExpandedAbbrev_250806fix": row[ColName_AE_Extraction_ExpandedAbbrev_250806fix], # expr setting 6
                    "Note_ExpandedAbbrev_250806fix": row[ColName_Note_ExpandedAbbrev_250806fix], # expr setting 7, 250806fix: full Notes with 250716 abbreviation expansion,
                    "GS_Extraction_ExpandedAbbrev_250806fix": row[ColName_GS_Extraction_ExpandedAbbrev_250806fix], # expr setting 8, 250806fix: Gold Standard Extraction with 250716 abbreviation expansion,
                    "gold_NIHSS": row[ColName_GoldStandard_NIHSS],
                    "gold_NIHSS_breakdown": row[ColName_GoldStandard_NIHSS_breakdown],
                } for index, row in df.iterrows()
                ]

                print(f"~!!!!length of the list of dictionaries: {len(output)},which should coincide with the number of rows in the CSV file")

                return output

            except KeyError:
                raise KeyError(f"At least one specified column is missing in the CSV file. Please ensure the CSV file contains the required columns: Sample_ID, {ColName_GoldStandardInput}, {ColName_GS_Extraction_ExpandedAbbrev}, {ColName_GS_Extraction_NoExpandedAbbrev}, {ColName_AE_Extraction_NoExpandedAbbrev}, {ColName_Note_ExpandedAbbrev}, {ColName_Note}, {ColName_AE_Extraction_ExpandedAbbrev_250806fix}, {ColName_Note_ExpandedAbbrev_250806fix}, {ColName_GS_Extraction_ExpandedAbbrev_250806fix}, {ColName_GoldStandard_NIHSS}, {ColName_GoldStandard_NIHSS_breakdown}")

        else:
            raise ValueError("Unsupported dataset format. Please provide a CSV file or a known benchmark.")
