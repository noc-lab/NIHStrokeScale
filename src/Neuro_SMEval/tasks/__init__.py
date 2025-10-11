def get_task(args):
    if args.task == 'aggregate_prompt':
        from Neuro_SMEval.tasks.task_AggrPrompt import AggrPrompt_Task
        print(f"args.task = {args.task}")
        return AggrPrompt_Task(args)
    elif args.task == 'aggregate_prompt_AllAblExpr' or args.task.lower().startswith('aggregate'):
        from Neuro_SMEval.tasks.task_AggrPrompt import AggrPrompt_Task_AllAblExpr
        print(f"args.task = {args.task}")
        return AggrPrompt_Task_AllAblExpr(args)
    else:
        raise NotImplementedError