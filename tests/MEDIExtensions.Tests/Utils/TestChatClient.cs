using Microsoft.Extensions.AI;

namespace MEDIExtensions.Tests.Utils;

/// <summary>
/// Callback-based <see cref="IChatClient"/> fake for unit testing.
/// Follows the dotnet/extensions TestChatClient pattern.
/// </summary>
public sealed class TestChatClient : IChatClient
{
    public IServiceProvider? Services { get; set; }

    public Func<IEnumerable<ChatMessage>, ChatOptions?, CancellationToken, Task<ChatResponse>>?
        GetResponseAsyncCallback { get; set; }

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
        => GetResponseAsyncCallback?.Invoke(messages, options, cancellationToken)
           ?? Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "default")));

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
        => throw new NotImplementedException("Streaming not used by MEDIExtensions components.");

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceType is not null && serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }

    /// <summary>
    /// Creates a TestChatClient that returns the given JSON string as the response text.
    /// </summary>
    public static TestChatClient WithJsonResponse(string json) => new()
    {
        GetResponseAsyncCallback = (_, _, _) =>
            Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, json)))
    };

    /// <summary>
    /// Creates a TestChatClient that throws the given exception.
    /// </summary>
    public static TestChatClient WithException(Exception ex) => new()
    {
        GetResponseAsyncCallback = (_, _, _) => throw ex
    };
}
