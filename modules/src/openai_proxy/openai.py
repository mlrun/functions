# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import json
from urllib.parse import urljoin
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request, Response, Body

app = FastAPI(
    title="OpenAI Proxy App",
    description="Local FastAPI proxy for OpenAI style endpoints",
    version="1.0.0",
)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")


def build_headers(incoming: dict) -> dict:
    headers = {}
    auth = incoming.get("authorization") or incoming.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    elif OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    ctype = incoming.get("content-type") or incoming.get("Content-Type") or "application/json"
    headers["Content-Type"] = ctype
    return headers


def build_target(path: str) -> str:
    base = OPENAI_BASE_URL
    if base.endswith("/v1") or base.endswith("/v1/"):
        base = base[:-3] if base.endswith("/v1") else base[:-4]
    return urljoin(base + "/", path.lstrip("/"))


def forward_json(path: str, body: dict, headers: dict, query: dict):
    target = build_target(path)
    resp = requests.post(
        target,
        headers=headers,
        params=query,
        json=body,
        timeout=60,
    )
    return resp

@app.get("/")
def health():
    return {"status": "ok"}


# relaxed chat endpoint, accepts any JSON that includes messages
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    if "messages" not in payload or not isinstance(payload["messages"], list):
        return Response(
            content=json.dumps({"error": "messages must be a list of chat messages"}),
            status_code=400,
            media_type="application/json",
        )

    if "model" not in payload or payload["model"] is None:
        payload["model"] = OPENAI_DEFAULT_MODEL

    headers = build_headers(dict(request.headers))
    resp = forward_json("/v1/chat/completions", payload, headers, dict(request.query_params))
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("Content-Type", "application/json"),
    )


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    if "model" not in payload or not payload["model"]:
        payload["model"] = "text-embedding-3-small"
    headers = build_headers(dict(request.headers))
    resp = forward_json("/v1/embeddings", payload, headers, dict(request.query_params))
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("Content-Type", "application/json"),
    )


@app.post("/v1/responses")
async def responses_api(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    if "model" not in payload or payload["model"] is None:
        payload["model"] = OPENAI_DEFAULT_MODEL
    headers = build_headers(dict(request.headers))
    resp = forward_json("/v1/responses", payload, headers, dict(request.query_params))
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("Content-Type", "application/json"),
    )


# ---------------- client ----------------
class OpenAIProxyClient:
    """
    Simple client for the local proxy.
    Default base url is http://localhost:8000
    If api_key is not provided, it uses OPENAI_API_KEY from environment.
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"messages": messages}
        if model:
            body["model"] = model
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def embeddings(self, text: Any, model: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"input": text}
        if model:
            body["model"] = model
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._headers(),
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def responses(self, input_text: Any, model: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"input": input_text}
        if model:
            body["model"] = model
        resp = requests.post(
            f"{self.base_url}/v1/responses",
            headers=self._headers(),
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()


# optional quick self test when running this file directly
if __name__ == "__main__":
    # start the server in another terminal first:
    # uvicorn openai_proxy.openai:app --host 0.0.0.0 --port 8000 --reload
    c = OpenAIProxyClient()
    try:
        print("Health:", requests.get(f"{c.base_url}/").json())
    except Exception as e:
        print("Server not running:", e)
