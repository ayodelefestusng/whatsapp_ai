
def assistant_node(state: State, config: RunnableConfig):
    # Fix for lint: "get" is not a known attribute of "None"
    if state is None:
        state = cast(State, {}) 
        
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    
    tenant_config = state.get("tenant_config", {}) or {}
    push_name = tenant_config.get("push_name", "User")
    
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("thread_id", "unknown")
    log_info(f"Assistant node triggered for tenant: {tenant_id}", tenant_id, conversation_id)

    # --- Resolve DB-sourced prompts with hardcoded fallbacks ---
    tenant_config = state.get("tenant_config") or {}
    employee_id = {state.get('employee_id')}
    current_year = datetime.now().year
    previous_year = current_year - 1
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    status_summary = state.get("status_summary", "No active application.")
    pdf_content = state.get("pdf_content", "None")
    web_content = state.get("web_content", "None")
    sql_result = state.get("sql_result", "None")

    # global_answer_prompt acts as the persona/intro section of system_prompt.
    # Falls back to GLOBAL_FINAL_ANSWER_PROMPT if not set in the DB.
    global_answer_prompt = (
        tenant_config.get("global_answer_prompt")
        or GLOBAL_FINAL_ANSWER_PROMPT
    )

    
    agent_prompt = tenant_config.get("agent_prompt") or DEFAULT_AGENT_PROMPT
    final_answer_prompt = tenant_config.get("final_answer_prompt") or DEFAULT_FINAL_ANSWER_PROMPT
    
    # Static system prompt instruction for pushName
    greeting_instruction = f"Always greet or address the user using their name: {push_name}, if they are starting a conversation or if appropriate."
    
    llm = get_llm_instance(tenant_id)
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state.get("messages", [])
    system_msg = SystemMessage(content=f"{agent_prompt}\n\n{greeting_instruction}\n\n{final_answer_prompt}")
    
    response = llm_with_tools.invoke([system_msg] + messages)
    
    if isinstance(response.content, str) and not response.tool_calls:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                t_name = parsed.get("name") or parsed.get("tool")
                if t_name:
                    response.tool_calls = [{"name": t_name, "args": parsed.get("args", {}), "id": str(uuid.uuid4()), "type": "tool_call"}]
            except: pass

    return {"messages": [response]}
