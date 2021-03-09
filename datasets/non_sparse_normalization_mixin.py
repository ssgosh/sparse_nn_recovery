from torchvision.transforms import transforms

from utils.torchutils import ClippedConstantTransform


class NonSparseNormalizationMixin:
    """Mixin class implementing different transforms for non-sparse versions of this dataset"""

    def get_train_transform_(self):
        return self._get_transforms_with_non_sparse(self.train_transforms)

    def get_test_transform_(self):
        return self._get_transforms_with_non_sparse(self.test_transforms)

    def _get_transforms_with_non_sparse(self, transforms_lst):
        t = list(transforms_lst)
        t.append(transforms.ToTensor())
        if self.non_sparse:
            t.append(ClippedConstantTransform(self.constant_pixel_val))
            t.append(
                transforms.Normalize(
                    (self.non_sparse_mean[self.constant_pixel_val],),
                    (self.non_sparse_std[self.constant_pixel_val],)
                )
            )
        else:
            t.append(transforms.Normalize((self.usual_mean,), (self.usual_std,)))
        return transforms.Compose(t)

    def get_without_transform_(self):
        if self.non_sparse:
            return transforms.Compose([
                transforms.ToTensor(),
                ClippedConstantTransform(self.constant_pixel_val)
            ])
        else:
            return transforms.ToTensor()

    def get_transformed_zero_one_mixin(self):
        if self.non_sparse:
            return self.non_sparse_zero[self.constant_pixel_val], self.non_sparse_one[self.constant_pixel_val]
        else:
            return self.usual_zero, self.usual_one

    def get_mean_std_mixin(self):
        if self.non_sparse:
            return self.non_sparse_mean[self.constant_pixel_val], self.non_sparse_std[self.constant_pixel_val]
        else:
            return self.usual_mean, self.usual_std

