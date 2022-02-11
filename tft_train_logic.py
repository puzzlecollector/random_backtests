# plot figures every epoch 
fig = plt.figure()
ax = fig.add_subplot(411)
ax1 = fig.add_subplot(412)
# ax2 = fig.add_subplot(413)
ax3 = fig.add_subplot(414)
plt.ion()


for i in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
    print('Training...')
    total_loss = 0 
    model.train() 
    
    for step, batch in tqdm(enumerate(train_dataloader)): 
        if step%200 == 0 and not step == 0: 
            model.eval() 
            val_loss, _, _ = evaluate(val_dataloader) 
            val_losses.append(val_loss) 
            if np.min(val_losses) == val_losses[-1]: 
                print("saving best checkpoint!") 
                torch.save(model.state_dict(), "TFT_binance_prototype.pt") 
            model.train()  
    
        
        past_cont = batch['past_cont'].to(device) 
        past_disc = batch['past_disc'].to(device) 
        target_seq = batch['target_seq'].to(device) 
        future_disc = batch['future_disc'].to(device) 
        model.reset(batch_size=past_cont.shape[0], gpu=True)
                
        optimizer.zero_grad() 

        net_out, vs_weights = model(x_past_cont = past_cont, 
                                    x_past_disc = past_disc, 
                                    x_future_cont = None, 
                                    x_future_disc = future_disc) 
        
        # net_out = net_out.cpu().detach()[0]
        #loss = torch.mean(QuantileLoss(net_out, target_seq, quantiles))  
        net_out = torch.reshape(net_out, (-1,1))
        loss = criterion(net_out, target_seq)
        # backward pass 
        train_losses.append(loss.item()) 
        loss.backward() 
        optimizer.step() 

        # loss graphs
        fig.tight_layout(pad=0.1) 
        ax.clear()
        ax.title.set_text("Training Loss")
        ax.plot(train_losses)
        
        ax1.clear() 
        ax1.title.set_text("Val Loss")
        ax1.plot(val_losses) 
        
        # visualise variable selection weights 
        vs_weights = torch.mean(torch.mean(vs_weights, dim=0), dim=0).squeeze()  
        vs_weights = vs_weights.cpu().detach().numpy()
        ax3.clear() 
        ax3.title.set_text("Variable Selection Weights") 
        plt.xticks(rotation=-30) 
        x = ["BTC Open", "BTC High", "BTC Low", "BTC Close", "BTC Volume",
            "ETH Open", "ETH High", "ETH Low", "ETH Close", "ETH Volume", "Hours"] 
        ax3.bar(x=x, height=vs_weights) 
        fig.canvas.draw() 
