'''
This file is used to run your functions and print the results
'''
import wandb
import numpy as np

# Import everything from functions.py file
from functions import train_loop, eval_loop, init_components, init_logger, get_num_parameters, get_args, save_model
from utils import init_data, DEVICE

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    lang, train_loader, val_loader, test_loader = init_data(args)

    slot_f1 = []
    int_acc = []
    if not args.test_only :
        for idx in range(1, args.runs+1) :
            model, optimizer, criterion_slots, criterion_intents = init_components(args, lang)
            
            if args.enable_logger :
                init_logger(args)
                tot_params, trainable_params = get_num_parameters(model)
                wandb.config.update({"model_size": tot_params})
                wandb.config.update({"trainable_size": trainable_params})
                wandb.config.update({"run_idx": idx})

            # model training
            best_model = train_loop(model, train_loader, val_loader, optimizer, criterion_slots, criterion_intents, lang,
                n_epochs=args.n_epochs,
                clip=5,
                patience=5
            ).to(DEVICE)

            # model inference
            results_test, intent_test, _, _ = eval_loop(best_model, test_loader, criterion_slots, criterion_intents, lang)    
            print(f"run {idx} : slot F1 = {results_test['total']['f']}, intent acc. = {intent_test['accuracy']}")
            slot_f1.append(results_test['total']['f'])
            int_acc.append(intent_test['accuracy'])

            # Save model weights
            if args.save_model :
                model_name = 'run-'+ str(idx) + '_model.pt'
                save_model(best_model, filename=model_name)

            # Close the logger
            if args.enable_logger:
                wandb.log({"mean Slot F1": results_test['total']['f'], "mean Intent accuracy": intent_test['accuracy']})
                wandb.finish()
                
        # average runs results and show
        mean_slot_f1 = np.asarray(slot_f1).mean()
        mean_int_acc = np.asarray(int_acc).mean()
        print('Slot F1: ', mean_slot_f1)
        print('Intent Accuracy:', mean_int_acc)
    
    # if test_only is enabled simply load the model and test it showing results
    else :
        model, optimizer, criterion_slots, criterion_intents = init_components(args, lang)
        # model inference
        results_test, intent_test, _, _ = eval_loop(model, test_loader, criterion_slots, criterion_intents, lang)    
        print(f"slot F1 = {results_test['total']['f']}, intent acc. = {intent_test['accuracy']}")
