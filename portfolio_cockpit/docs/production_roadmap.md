# Production Roadmap

## Phase 1: Local MVP

Capabilities:

- Read the professor's Excel workbook.
- Normalize trades into SQLite.
- Calculate positions, valuation, returns, risk metrics and reports.
- Add manual trades and import NEW TRADES without changing Excel.

Risks:

- Simplified validation rules.
- Local price cache only.
- Basic cash reconstruction.
- SQLite database stored on one machine.

Technical improvements:

- Expand tests around every workbook interpretation rule.
- Add structured error logging.
- Create reproducible demo reset scripts.

Governance improvements:

- Document the operating procedure.
- Make the data-quality tab the required first control.

Testing and reconciliation needs:

- Reconcile positions against the Excel case and broker-like statements.

## Phase 2: Robust Local Tool

Capabilities:

- Versioned input folder for immutable raw Excel files.
- Scheduled SQLite backups.
- Stronger validation and exception handling.
- Local packaging as a desktop-like app.

Risks:

- Manual file handling can still create version confusion.
- Secrets may be misconfigured on a user's PC.

Technical improvements:

- Add migration scripts for SQLite schema changes.
- Add backup and restore commands.
- Add market-data refresh from approved local cache or provider.

Governance improvements:

- Establish naming rules for input files.
- Maintain an audit trail for imports, resets and reports.

Testing and reconciliation needs:

- Daily reconciliation of positions, cash and realized P&L.

## Phase 3: Professional Internal Tool

Capabilities:

- Broker or custodian reconciliation.
- Market-data provider integration.
- Benchmark data refresh.
- Controlled secrets management for Gemini API keys.
- Role-based access for internal users.

Risks:

- Vendor data gaps.
- Incorrect mapping between operational symbols and market-data symbols.
- LLM privacy exposure if confidential data is sent to external services.

Technical improvements:

- Add access control.
- Centralize logs.
- Use environment-managed secrets.
- Add integration tests against sample broker and market-data files.

Governance improvements:

- Define exception approval workflow.
- Define LLM prompt policy and prohibit confidential raw transactions in prompts.

Testing and reconciliation needs:

- Automated broker, custodian and benchmark reconciliation.

## Phase 4: Controlled Production Deployment

Capabilities:

- Deployed internal app with backups, monitoring, access control and recovery plan.
- Formal release process.
- Documented support model.
- A future robust version could evaluate Docker plus MariaDB or PostgreSQL for team deployment, but that is intentionally out of scope for this classroom MVP.

Risks:

- Operational dependency on a dashboard if controls are weak.
- Regulatory or confidentiality issues if LLM usage is not governed.

Technical improvements:

- Add deployment pipeline.
- Add monitoring and alerting.
- Add disaster recovery testing.
- Add automated security scanning.

Governance improvements:

- Formal data ownership.
- Change approval.
- Incident response procedure.
- Periodic access review.

Testing and reconciliation needs:

- Regression suite for metrics.
- Daily production reconciliation.
- Periodic recovery drills.
