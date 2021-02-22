from contextlib import contextmanager

# Saves and restores requires_grad and training variables appropriately for the model
# Within context, model is set to eval and requires_grad is False for all model parameters.

@contextmanager
def model_eval_no_grad(model):
    try:
        model_was_training = model.training
        if model_was_training:
            model.eval()

        requires_grad_dict = _save_requires_grad_and_set_to_false(model)
        yield
    finally:
        _restore_requires_grad(model, requires_grad_dict)
        if model_was_training:
            model.train()

@contextmanager
def images_require_grad(images):
    try:
        images_required_grad = images.requires_grad
        images.requires_grad = True
        yield
    finally:
        images.requires_grad = images_required_grad

def _save_requires_grad_and_set_to_false(model):
    # Save requires gradient
    requires_grad_dict = {}
    for name, param in model.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad = False
    return requires_grad_dict


def _restore_requires_grad(model, requires_grad_dict):
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_dict[name]


