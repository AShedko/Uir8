def visualize_model(model, num_images=6):
# bookkeeping and setup of figues
    was_training = model.training # keep the previous state of model
    model.eval() # set eval mode (BatchNorm and Dropout work differently)
    images_so_far = 0
    fig = plt.figure()
​
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
	# Move inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
​
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
​
            for j in range(inputs.size()[0]):
	    # Draw Images and predictions
                images_so_far += 1
                ax = plt.subplot(num_images//3, 3, images_so_far)
                ax.axis('off')
                ax.set_title(
                  'predicted: {}\n actual: {}'.
                  format(class_names[preds[j]],
                         class_names[labels[j]]))
                imshow(inputs.cpu().data[j])
​
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
