from contextlib import contextmanager


@contextmanager
def model_eval_no_grad(model):
    try:
        model_was_training = model.training
        if model_was_training:
            model.eval()

        requires_grad = _save_requires_grad(model)
        yield
    finally:
        _restore_requires_grad(model, requires_grad)
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

def _save_requires_grad(model):
    # Save requires gradient
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad = False
    return requires_grad


def _restore_requires_grad(model, requires_grad):
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad[name]
        param.requires_grad = False


