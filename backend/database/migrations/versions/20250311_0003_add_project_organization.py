"""add organization to projects

Revision ID: 20250311_0003
Revises: 20250311_0002
Create Date: 2025-03-11 00:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20250311_0003"
down_revision = "20250311_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "projects" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("projects")}
    if "organization" not in columns:
        op.add_column("projects", sa.Column("organization", sa.String(length=255), nullable=True))
        op.execute("UPDATE projects SET organization = 'default'")
        op.alter_column("projects", "organization", nullable=False)
        inspector = sa.inspect(bind)

    unique_constraints = {uc["name"] for uc in inspector.get_unique_constraints("projects")}
    if "uq_projects_project_key" in unique_constraints:
        op.drop_constraint("uq_projects_project_key", "projects", type_="unique")
        inspector = sa.inspect(bind)

    unique_constraints = {uc["name"] for uc in inspector.get_unique_constraints("projects")}
    if "uq_projects_org_key" not in unique_constraints:
        op.create_unique_constraint("uq_projects_org_key", "projects", ["organization", "project_key"])
        inspector = sa.inspect(bind)

    indexes = {idx["name"] for idx in inspector.get_indexes("projects")}
    if "ix_projects_organization" not in indexes:
        op.create_index("ix_projects_organization", "projects", ["organization"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_projects_organization", table_name="projects")
    op.drop_constraint("uq_projects_org_key", "projects", type_="unique")
    op.create_unique_constraint("uq_projects_project_key", "projects", ["project_key"])
    op.drop_column("projects", "organization")
