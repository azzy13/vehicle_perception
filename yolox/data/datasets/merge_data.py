from .datasets_wrapper import Dataset

class MergeDataset(Dataset):
    def __init__(
        self,
        datasets,
        img_size=(608, 1088),
        preproc=None,
    ):
        super().__init__(img_size)
        self.datasets = datasets
        self.img_size = img_size
        self.preproc = preproc
        self.img_len = 0
        for dataset in self.datasets:
            self.img_len += len(dataset)

    def __len__(self):
        return self.img_len

    def compute_dataset_and_index(self, index):
        current_position = 0
        for i in range(len(self.datasets)):
            new_index = index - current_position
            if (new_index < len(self.datasets[i])):
                return i, new_index
            current_position += len(self.datasets[i])
        return -1, -1

    def pull_item(self, index):
        dataset_selection, new_index = self.compute_dataset_and_index(index)
        dataset = self.datasets[dataset_selection]
        return dataset.pull_item(new_index)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)
        initial_size = img.shape

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, int(img_id)
