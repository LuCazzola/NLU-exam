'''
This file is used to run your functions and print the results
'''
import wandb
import numpy as np

# Import everything from functions.py file
from functions import train_loop, eval_loop, get_args, get_num_parameters, init_components, init_logger
from utils import init_data, DEVICE

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    if args.enable_logger :
        init_logger(args)

    lang, train_loader, val_loader, test_loader = init_data(args)

    slot_f1 = []
    int_acc = []
    for idx in range(1, args.runs+1) :
        model, optimizer, criterion_slots, criterion_intents = init_components(args, lang)
        
        if idx == 1 :
            tot_params, trainable_params = get_num_parameters(model)
            print(f"Number of parameters: {tot_params}")
            print(f"Trainable parameters: {trainable_params}")

            if args.enable_logger :
                wandb.config.update({"model_size": tot_params})
                wandb.config.update({"trainable_size": tot_params})

        # model training
        best_model = train_loop(model, train_loader, val_loader, optimizer, criterion_slots, criterion_intents, lang,
            n_epochs=args.n_epochs,
            clip=5,
            patience=5
        ).to(DEVICE)

        # model inference
        results_test, intent_test, _ = eval_loop(model, test_loader, criterion_slots, criterion_intents, lang)    
        print(f"run {idx} : slot F1 = {results_test['total']['f']}, intent acc. = {intent_test['accuracy']}")
        slot_f1.append(results_test['total']['f'])
        int_acc.append(intent_test['accuracy'])

    
    # average runs results and show
    mean_slot_f1 = np.asarray(slot_f1).mean()
    mean_int_acc = np.asarray(int_acc).mean()
    print('Slot F1: ', mean_slot_f1)
    print('Intent Accuracy:', mean_int_acc)

    # Close the logger
    if args.enable_logger:
        wandb.log({"mean Slot F1": mean_slot_f1, "mean Intent accuracy": mean_int_acc})
        wandb.finish()


