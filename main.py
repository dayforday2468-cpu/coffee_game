from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def create_customer(llm: ChatOllama) -> str:
    """카페에 방문한 가상의 손님을 생성한다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                우리 팀은 커피 내기 게임을 하려고 합니다.
                당신은 카페에 방문한 손님입니다.

                현재 기분, 취향, 상황을 포함해 손님을 100자 이내로 묘사하세요.
                특정 메뉴를 직접 말하지는 마세요.
                """,
            )
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({}).strip()


def get_menu_recommendations(num_players: int) -> list[str]:
    """각 참가자의 추천 메뉴를 입력받는다."""

    recommendations = []

    for player_number in range(1, num_players + 1):
        while True:
            menu = input(
                f"{player_number}번 참가자가 추천할 메뉴를 입력하세요: "
            ).strip()

            if menu:
                recommendations.append(menu)
                break

            print("메뉴를 한 글자 이상 입력해주세요.")

    return recommendations


def select_winner(
    llm: ChatOllama,
    situation: str,
    menu_recommendations: list[str],
) -> str:
    """손님의 상황을 기반으로 가장 적절한 메뉴를 추천한 참가자를 고른다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 앞에서 설정된 카페 손님입니다.

                손님의 상황과 참가자들이 추천한 메뉴를 보고,
                지금 가장 먹기 싫은 메뉴 하나를 선택하세요.

                반드시 한 명만 선택해야 합니다.
                메뉴를 추천한 참가자의 번호를 정확히 표시하세요.

                다음 형식으로만 답하세요.

                선택된 참가자: N번
                선택한 메뉴: 메뉴 이름
                선택 이유: 한 문장
                """,
            ),
            (
                "human",
                """
                손님의 상황:
                {situation}

                참가자별 추천 메뉴:
                {recommendations}
                """,
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    recommendations_text = "\n".join(
        f"{number}번 참가자: {menu}"
        for number, menu in enumerate(menu_recommendations, start=1)
    )

    return chain.invoke(
        {
            "situation": situation,
            "recommendations": recommendations_text,
        }
    ).strip()


def main() -> None:
    llm = ChatOllama(
        model="llama3",
        temperature=0.7,
    )

    situation = create_customer(llm)

    print("\n=== 오늘의 손님 ===")
    print(situation)

    while True:
        try:
            num_players = int(input("\n게임에 참여할 사람 수는? "))

            if num_players >= 2:
                break

            print("참가자는 2명 이상이어야 합니다.")

        except ValueError:
            print("숫자를 입력해주세요.")

    menu_recommendations = get_menu_recommendations(num_players)

    result = select_winner(
        llm=llm,
        situation=situation,
        menu_recommendations=menu_recommendations,
    )

    print("\n=== 선택 결과 ===")
    print(result)


if __name__ == "__main__":
    main()