# ibkr-mcp

Interactive Brokers MCP Server for AI-powered trading assistance.

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that connects AI assistants to Interactive Brokers, enabling intelligent portfolio management, options analysis, and risk monitoring.

## Features

- **Account Management**: Real-time account summary, portfolio positions, and P&L tracking
- **Options Analysis**: Option chain fetching, Greeks calculation, and strategy scanning
- **Risk Monitoring**: Portfolio risk evaluation with configurable limits and alerts
- **Trading Strategies**: Built-in support for covered calls, iron condors, PMCC, vertical spreads
- **News Integration**: Historical news retrieval for market research
- **Playbook Actions**: Automated adjustment suggestions based on risk rules

## Installation

```bash
pip install ibkr-mcp
```

## Prerequisites

- Python 3.12+
- Interactive Brokers TWS or IB Gateway running
- Valid IBKR account credentials

## Configuration

Set environment variables or create a `.env` file:

```bash
# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=4001          # TWS: 7497, IB Gateway: 4001
IBKR_CLIENT_ID=0
IBKR_ACCOUNT=           # Optional: specific account ID

# Optional: Data directories
IBKR_MCP_OPTION_DATA_DIR=optiondata
IBKR_MCP_OPTION_HISTORY_DIR=historydata
IBKR_MCP_MARKET_DATA_TYPE=LIVE  # LIVE, FROZEN, DELAYED, DELAYED_FROZEN
```

## Usage

### As MCP Server

Add to your Claude Desktop or other MCP-compatible client configuration:

```json
{
  "mcpServers": {
    "ibkr-mcp": {
      "command": "uvx",
      "args": [
        "ibkr-mcp"
      ],
      "env": {
        "IBKR_ACCOUNT": "U1234567"
      }
    }
  }
}
```

### Available Tools

| Tool                          | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| `get_account_summary`       | Retrieve account summary for specific or all accounts  |
| `get_portfolio`             | Get portfolio positions with P&L                       |
| `get_positions`             | Get normalized positions across accounts               |
| `get_greeks_summary`        | Calculate portfolio Greeks (delta, gamma, theta, vega) |
| `get_option_chains`         | Fetch option chain snapshots                           |
| `scan_option_signals`       | Scan for options strategy trade signals                |
| `evaluate_portfolio_risk`   | Evaluate risk against configured limits                |
| `generate_playbook_actions` | Generate adjustment suggestions                        |
| `get_historical_news`       | Retrieve historical news for symbols                   |

### Example Queries

Once connected through an AI assistant, you can ask:

- "What's my current account balance and buying power?"
- "Show me all my positions with Greeks"
- "What's the option chain for AAPL expiring next month?"
- "Evaluate my portfolio risk and suggest adjustments"
- "Find covered call opportunities in my portfolio"

## Risk Configuration

Create a `risk.yaml` file to define risk limits:

```yaml
limits:
  max_delta: 100
  max_theta: -500
  max_concentration: 0.25

roll_rules:
  min_dte: 7
  target_delta: 0.30
```

## Support

For issues and feature requests, please contact the maintainers.

## License

Proprietary - All rights reserved.

## Disclaimer

This software is for informational purposes only. It does not constitute financial advice. Trading involves substantial risk of loss. Use at your own risk.
