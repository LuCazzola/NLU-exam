"""
Module for running functions and printing the results.
"""
import wandb
# import functions to initialize / train / evaluate the model
from functions import init_modelComponents, init_data, init_logger, train_loop, eval_loop, get_args, save_model, get_num_parameters
# inmport components to handle the dataset
from utils import init_data, DEVICE

if __name__ == "__main__":
    # Parse command line inputs
    args = get_args()
    # Initialize WanDB logger
    if args.enable_logger:
        init_logger(args)
    
    # Initialize components
    lang, train_loader, val_loader, test_loader = init_data(args)
    model, optimizer, criterion_train, criterion_eval = init_modelComponents(args, lang)

    if args.enable_logger :
        tot_params, trainable_params = get_num_parameters(model)
        wandb.config.update({"model_size": tot_params})
        wandb.config.update({"trainable_size": trainable_params})

    # if the flag is unset perform Train + Testing, otherwise perform test only
    if not args.test_only :
        # Train the model
        best_model = train_loop(model, train_loader, val_loader, optimizer, criterion_train, criterion_eval, n_epochs=args.n_epochs, patience=3, clip=5)
        best_model.to(DEVICE)
    else :
        best_model = model
    
    # Evaluate the best model found
    ppl_test,  _ = eval_loop(best_model, test_loader, criterion_eval)
    print('Test ppl: ', ppl_test)

    # Save model weights
    if args.save_model :
        save_model(best_model, filename='best_model.pt')
    # Close the logger
    if args.enable_logger:
        wandb.log({"test perplexity": ppl_test})
        wandb.finish()

