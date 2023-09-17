use warp::{Filter, Rejection, Reply};
use serde::{Deserialize, Serialize};
use std::error::Error;
use async_openai::{
    types::{ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs, Role},
    Client,
};
use async_openai::config::OpenAIConfig;

#[derive(Debug, Deserialize, Serialize)]
struct PingResponse {
    status: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAiRequest {
    api_key: String,
    question: String
}

#[derive(Debug, Deserialize, Serialize)]
struct ApiResponse {
    answer: String
}

async fn get_response(request: OpenAiRequest) -> Result<String, Box<dyn Error>> {
    let config = OpenAIConfig::new()
        .with_api_key(request.api_key);

    let client = Client::with_config(config);

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u16)
        .model("gpt-3.5-turbo-0613")
        .messages([ChatCompletionRequestMessageArgs::default()
            .role(Role::User)
            .content(request.question)
            .build()?])
        .build()?;

    let response = client
        .chat()
        .create(request)
        .await?;

    // Assuming response.choices contains at least one choice
    if let Some(choice) = response.choices.first() {
        let content = choice.message.content.clone().expect("REASON").to_string(); // Convert to String
        return Ok(content);
    }

    Err("No response from GPT-3.5 Turbo".into())
}

async fn ping_handler() -> Result<impl Reply, Rejection> {
    let response = PingResponse {
        status: "ok".to_owned(),
    };
    Ok(warp::reply::json(&response))
}

async fn answer_handler(request: OpenAiRequest) -> Result<impl Reply, Rejection> {
    let response = match get_response(request).await {
        Ok(answer) => ApiResponse { answer },
        Err(e) => ApiResponse { answer: e.to_string() },
    };

    Ok(warp::reply::json(&response))
}


#[tokio::main]
async fn main() {
    let ping_route = warp::path("ping")
        .and(warp::get())
        .and_then(ping_handler);

    let answer_route = warp::path("answer")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(answer_handler);

    let routes = answer_route.or(ping_route);

    warp::serve(routes)
        .run(([127, 0, 0, 1], 8080))
        .await;
}
