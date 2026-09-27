"""The model catalog: what each provider serves, and what logical tiers resolve to.

A tier is a logical model name that resolves to a different physical model per provider —
``chat-fast`` might mean ``openai/gpt-oss-20b`` on Groq and ``llama3.2`` on Ollama. This is
what makes a fallback chain able to actually serve a fallback: without it, the providers in
a chain serve disjoint model sets, so every provider after the first rejects the request
and the chain is decorative.

A tier declares only the **mapping**, never the order. Ordering stays with the team's
``provider_priority``, so exactly one place decides which provider is preferred. Letting a
tier express preference too would give two sources for the same decision, and the
interesting question during an incident — why did this request go there? — would have two
possible answers.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCatalog:
    """Which providers can serve which models, directly or through a tier."""

    #: provider name -> the physical models it serves
    providers: dict[str, set[str]] = field(default_factory=dict)
    #: tier name -> {provider name: physical model to send that provider}
    tiers: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve(self, model: str) -> dict[str, str]:
        """Return ``{provider: physical_model}`` for a requested model name.

        Handles both kinds of request through one path. A tier name resolves to its
        declared mapping; any other name is treated as a physical model, which resolves to
        every provider that serves it mapped to itself. Callers therefore never need to
        know which kind they were given, and a physical model name keeps working exactly as
        it did before tiers existed.
        """
        if model in self.tiers:
            return dict(self.tiers[model])

        return {
            provider_name: model
            for provider_name, served in self.providers.items()
            if model in served
        }

    def is_tier(self, model: str) -> bool:
        return model in self.tiers

    def physical_models_for(self, model: str) -> set[str]:
        """Every physical model a request for ``model`` could end up billed against.

        A budget reservation has to cover the worst case, and with a tier the price is not
        known until routing picks a provider.
        """
        return set(self.resolve(model).values())
