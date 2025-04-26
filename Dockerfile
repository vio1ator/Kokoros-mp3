# syntax=docker/dockerfile:1
FROM rust:1.86.0-slim-bookworm AS builderrs

RUN apt-get update -qq && apt-get install -qq -y wget pkg-config libssl-dev clang git cmake && rustup component add rustfmt

WORKDIR /app

COPY . .
COPY Cargo.toml .
COPY Cargo.lock .

RUN cargo build --release
RUN chmod +x ./download_all.sh && ./download_all.sh

FROM debian:sid-slim AS runner

WORKDIR /app

COPY --from=builderrs /app/target/release/build ./build
COPY --from=builderrs /app/target/release/koko ./koko
COPY --from=builderrs /app/data ./data
COPY --from=builderrs /app/checkpoints ./checkpoints

RUN chmod +x ./koko && apt-get update -qq && apt-get install -qq -y pkg-config libssl-dev 

EXPOSE 3000

ENTRYPOINT [ "./koko" ]
