
- mapa -> obrazek


When training:

Monitor Key Metrics:
- Episode Return/Reward: The primary indicator of performance. Is it increasing?
- Losses: Value loss, policy loss, reward loss (for the model), and SSL loss. Are they decreasing? Are they stable or fluctuating wildly?
- Exploration: How many unique states is the agent visiting? (This might require custom logging).


I set up tensorboard and I am looking at my metrics now. There are these tabs:
- Buffer
- collector_iter
- collector_step
- evaluator_iter
- evaluator_step
- learner_iter
- learner_step




