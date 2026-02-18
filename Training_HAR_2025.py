def main():
    # Configuration
    config = {
        'num_classes': 60,  # NTU-RGBD classes
        'num_joints': 25,
        'num_frames': 300,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'hidden_dim': 256,
        'num_heads': 8
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_transform = SkeletonTransform(apply_augmentation=True)
    test_transform = SkeletonTransform(apply_augmentation=False)
    
    train_dataset = SkeletonDataset(
        'data/ntu_train_skeletons.json',
        'data/ntu_train_labels.json',
        transform=train_transform
    )
    
    test_dataset = SkeletonDataset(
        'data/ntu_test_skeletons.json', 
        'data/ntu_test_labels.json',
        transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = EMS_TAGCN(
        num_classes=config['num_classes'],
        num_joints=config['num_joints'],
        num_frames=config['num_frames'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads']
    )
    
    # Create trainer
    trainer = HAR_Trainer(model, device, config['num_classes'])
    trainer.setup_optimizer(config['learning_rate'])
    
    # Training loop
    best_accuracy = 0
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        test_acc, predictions, labels = trainer.evaluate(test_loader)
        
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, 'best_model.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        print('-' * 50)
    
    print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()