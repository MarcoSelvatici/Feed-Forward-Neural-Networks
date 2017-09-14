import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

INITIAL_EPISODES = 5000
FINAL_EPISODES = 10
MINIMUM_SCORE = 50

class DL_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # input layer of size self.state_size and first hidden layer of size 5
        model.add(Dense(5, input_dim=self.state_size, activation='relu'))
        # second hidden layer of size 5
        model.add(Dense(5, activation='relu'))
        # output layer of size self.action_size
        model.add(Dense(self.action_size, activation='linear'))

        # create the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        state = np.reshape(state, (1, self.state_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # example of what I need to feed the net with:
    # - state --> bidimensional np.array of shape (1, state_size)
    # [[-0.00088245 - 0.0269204 - 0.04669272  0.02080927]]
    # - target --> bidimensional np.array of shape (1, action_size)
    # [[1.  0.]]
    def train(self, x, y):
        print("====== Training =======")
        for i in range(len(x)):
            tmp_x = np.reshape(x[i], (1, self.state_size))
            tmp_y = np.reshape(y[i], (1, self.action_size))
            self.model.fit(tmp_x, tmp_y, epochs=1, verbose=0)

    def get_training_samples(self, env):
        x = []
        y = []
        for e in range(INITIAL_EPISODES):
            game_x = []
            game_y = []
            state = env.reset()
            score = 0
            for _ in range(500):
                action = env.action_space.sample()
                game_x.append(state)
                # I need a target like [a, b, c, ... , z], so I have to
                # translate the action in a "one-hot" array
                if action == 0:
                    game_y.append([1.0, 0.0])
                else:
                    game_y.append([0.0, 1.0])
                state, reward, done, _ = env.step(action)
                score += reward

                if done:
                    if score > MINIMUM_SCORE:
                        for i in game_x:
                            x.append(i)
                        for i in game_y:
                            y.append(i)
                    break
        return x, y

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DL_agent(state_size, action_size)

    x, y = agent.get_training_samples(env)
    agent.train(x, y)

    for e in range(FINAL_EPISODES):
        state = env.reset()
        score = 0
        for _ in range(500):
            env.render()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                print("episode: {}, score: {}".format(e, score))
                break

if __name__ == "__main__":
    main()