use warp::{Filter, Rejection, Reply};
use serde::{Deserialize, Serialize};
use std::error::Error;
use async_openai::{
    types::{CreateChatCompletionRequestArgs, Role},
    Client,
};
use async_openai::config::OpenAIConfig;
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionResponse};
use tokio::time::timeout;

const DEFAULT_TIMEOUT: u64 = 120;

#[derive(Debug, Deserialize, Serialize)]
struct PingResponse {
    status: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: Role,
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAiRequest {
    api_key: String,
    model: String,
    max_tokens: u16,
    temperature: f32,
    timeout: Option<u64>,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ApiResponse {
    success: bool,
    openai_answer: Option<CreateChatCompletionResponse>,
    error: Option<String>,
}

async fn get_response(request: OpenAiRequest) -> Result<CreateChatCompletionResponse, Box<dyn Error>> {
    let req_timeout = match request.timeout {
        Some(x) => x,
        _ => DEFAULT_TIMEOUT
    };
    let duration = tokio::time::Duration::from_secs(req_timeout);

    let config = OpenAIConfig::new().with_api_key(request.api_key);

    let client = Client::with_config(config);

    if request.messages.is_empty()  {
        return Err("No response from GPT-3.5 Turbo".into())
    }

    let converted_messages: Vec<ChatCompletionRequestMessage> = request.messages
    .iter()
    .map(|m| ChatCompletionRequestMessage {
        role: m.role.clone(),
        content: Some(m.content.clone()),
        name: None,
        function_call: None,
    }) // Build each message inside the map
    .collect::<Vec<ChatCompletionRequestMessage>>();

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(request.max_tokens)
        .model(request.model)
        .temperature(request.temperature)
        .messages(converted_messages)
        .build()?;

    let task = async {
        client.chat().create(request).await
    };
    let result = timeout(duration, task).await?;

    return match result {
        Ok(x) => {
            Ok(x)
        }
        Err(e) => { Err(e.into()) }
    }
}

async fn ping_handler() -> Result<impl Reply, Rejection> {
    let response = PingResponse {
        status: "ok".to_owned(),
    };
    Ok(warp::reply::json(&response))
}

async fn answer_handler(request: OpenAiRequest) -> Result<impl Reply, Rejection> {
    let response = match get_response(request).await {
        Ok(answer) => ApiResponse {
            success: true,
            openai_answer: Some(answer),
            error: None
        },
        Err(e) => ApiResponse {
            success: false, openai_answer: None, error: Some(e.to_string())
        },
    };

    Ok(warp::reply::json(&response))
}


#[tokio::main]
async fn main() {
    std::env::set_var("RUST_LOG", "warn");

    let ping_route = warp::path("ping")
        .and(warp::get())
        .and_then(ping_handler);

    let answer_route = warp::path("answer")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(answer_handler);

    let routes = answer_route.or(ping_route);

    warp::serve(routes)
        .run(([0, 0, 0, 0], 8080))
        .await;
}
