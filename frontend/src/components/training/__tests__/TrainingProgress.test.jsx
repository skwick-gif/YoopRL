import React from 'react';
import { render, waitFor } from '@testing-library/react';
import TrainingProgress from '../TrainingProgress';

jest.mock('../../../services/trainingAPI', () => ({
  getProgress: jest.fn(),
  stopTraining: jest.fn()
}));

const { getProgress } = require('../../../services/trainingAPI');

window.alert = jest.fn();

describe('TrainingProgress', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('notifies parent when training completes', async () => {
    const onTrainingStatusChange = jest.fn();

    getProgress.mockResolvedValue({
      success: true,
      data: {
        status: 'completed',
        progress: 100,
        current_episode: 10,
        total_episodes: 10,
        avg_reward: 1.23,
        recent_loss: 0.01
      }
    });

    render(
      <TrainingProgress
        isDownloading={false}
        isTraining
        downloadProgress={0}
        trainingProgress={0}
        dataDownloaded
        downloadMessage=""
        trainingId="session-123"
        handleDownloadData={() => {}}
        handleStartTraining={() => {}}
        onTrainingStatusChange={onTrainingStatusChange}
        onTrainingStopped={() => {}}
      />
    );

    await waitFor(() => expect(getProgress).toHaveBeenCalled());

    await waitFor(() => {
      expect(onTrainingStatusChange).toHaveBeenCalledWith({ status: 'completed', progress: 100 });
    });
  });
});
