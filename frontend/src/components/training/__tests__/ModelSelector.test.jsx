import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import ModelSelector from '../ModelSelector';

jest.mock('../../../services/trainingAPI', () => ({
  loadModels: jest.fn()
}));

const { loadModels } = require('../../../services/trainingAPI');

describe('ModelSelector', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('keeps SAC models visible when agentType is SAC_INTRADAY_DSR', async () => {
    loadModels.mockResolvedValue({
      success: true,
      models: [
        {
          model_id: 'sac_TNA_v1',
          agent_type: 'SAC',
          symbol: 'TNA',
          version: 'v1',
          created_at: '2025-01-01T00:00:00Z',
          sharpe_ratio: 1.57,
          total_return: 12.34,
          episodes: 5000
        },
        {
          model_id: 'ppo_AAPL_v1',
          agent_type: 'PPO',
          symbol: 'AAPL',
          version: 'v1',
          created_at: '2025-01-01T00:00:00Z',
          sharpe_ratio: 1.42,
          total_return: 10.12,
          episodes: 4000
        }
      ]
    });

    render(<ModelSelector agentType="SAC_INTRADAY_DSR" onModelSelect={() => {}} />);

    const select = await screen.findByRole('combobox');

    await waitFor(() => expect(loadModels).toHaveBeenCalled());

    const optionLabels = Array.from(select.options).map((option) => option.textContent || '');

    expect(optionLabels.some((label) => label.includes('SAC - TNA'))).toBe(true);
    expect(optionLabels.some((label) => label.includes('PPO - AAPL'))).toBe(false);
  });
});
